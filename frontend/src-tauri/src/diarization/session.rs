// diarization/session.rs
//
// Per-recording diarization state: embedding extractor + online clusterer.
// Created when a recording starts (if the feature is enabled and the model
// is present) and dropped when it ends.

use super::clustering::{SpeakerClusterer, SpeakerClusteringConfig, cosine_similarity};
use super::embedding::{EmbeddingError, EmbeddingExtractor};
use super::timeline::{RollingDiarizationBuffer, SpeakerTimeline, SpeakerTimelineSegment};
use std::path::Path;

/// Minimum samples needed for the fbank frontend to produce the 10 frames
/// required by EmbeddingExtractor::compute (25ms frame + 9 * 10ms shifts).
const MIN_SAMPLES_FOR_EMBEDDING: usize = 1_840;
const DEFAULT_DIARIZATION_WINDOW_SECONDS: f64 = 10.0;
const DEFAULT_DIARIZATION_STRIDE_SECONDS: f64 = 5.0;
const DIARIZATION_SAMPLE_RATE: u32 = 16_000;

pub const DEFAULT_MIN_RELIABLE_SEGMENT_MS: u32 =
    (MIN_SAMPLES_FOR_EMBEDDING as u32 * 1000) / DIARIZATION_SAMPLE_RATE;

#[derive(Debug, Clone, Copy)]
pub struct DiarizationSessionConfig {
    pub model_id: &'static str,
    pub clustering: SpeakerClusteringConfig,
    pub min_reliable_segment_ms: u32,
}

impl Default for DiarizationSessionConfig {
    fn default() -> Self {
        Self {
            model_id: super::models::DEFAULT_EMBEDDING_MODEL_ID,
            clustering: SpeakerClusteringConfig::default(),
            min_reliable_segment_ms: DEFAULT_MIN_RELIABLE_SEGMENT_MS,
        }
    }
}

#[cfg(test)]
fn has_enough_samples_for_embedding(samples_len: usize) -> bool {
    samples_len >= MIN_SAMPLES_FOR_EMBEDDING
}

fn select_live_speaker_label(
    timeline_label: Option<String>,
    direct_segment_label: Option<String>,
    last_label: Option<String>,
) -> Option<String> {
    direct_segment_label.or(timeline_label).or(last_label)
}

#[cfg(test)]
fn should_compute_direct_segment_label(samples_len: usize) -> bool {
    has_enough_samples_for_embedding(samples_len)
}

fn has_enough_samples_for_reliable_segment(samples_len: usize, min_segment_ms: u32) -> bool {
    let min_samples = ((min_segment_ms as usize) * DIARIZATION_SAMPLE_RATE as usize / 1000)
        .max(MIN_SAMPLES_FOR_EMBEDDING);
    samples_len >= min_samples
}

#[derive(Clone)]
struct OfflineEmbedding {
    segment_index: usize,
    start_time: f64,
    end_time: f64,
    embedding: Vec<f32>,
}

#[derive(Clone)]
struct OfflineCluster {
    members: Vec<usize>,
    centroid: Vec<f32>,
}

fn normalized_weighted_centroid(
    a: &[f32],
    a_weight: usize,
    b: &[f32],
    b_weight: usize,
) -> Vec<f32> {
    let total = (a_weight + b_weight) as f32;
    let mut centroid = a
        .iter()
        .zip(b.iter())
        .map(|(a, b)| ((*a * a_weight as f32) + (*b * b_weight as f32)) / total)
        .collect::<Vec<_>>();
    let norm = centroid.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut centroid {
            *value /= norm;
        }
    }
    centroid
}

fn offline_clusters(
    records: &[OfflineEmbedding],
    config: SpeakerClusteringConfig,
) -> Vec<OfflineCluster> {
    let mut clusters = records
        .iter()
        .enumerate()
        .map(|(record_index, record)| OfflineCluster {
            members: vec![record_index],
            centroid: record.embedding.clone(),
        })
        .collect::<Vec<_>>();

    // ponytail: O(n^3) is fine for import chunks; use a priority queue if huge imports get slow.
    loop {
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let similarity = cosine_similarity(&clusters[i].centroid, &clusters[j].centroid);
                if best.map_or(true, |(_, _, best_similarity)| similarity > best_similarity) {
                    best = Some((i, j, similarity));
                }
            }
        }

        let Some((i, j, similarity)) = best else {
            break;
        };

        if similarity < config.cluster_similarity_threshold
            && clusters.len() <= config.max_anonymous_speakers
        {
            break;
        }

        let right = clusters.remove(j);
        let left = &mut clusters[i];
        let left_len = left.members.len();
        let right_len = right.members.len();
        left.centroid =
            normalized_weighted_centroid(&left.centroid, left_len, &right.centroid, right_len);
        left.members.extend(right.members);
    }

    clusters
}

pub struct DiarizationSession {
    extractor: EmbeddingExtractor,
    clusterer: SpeakerClusterer,
    rolling_buffer: RollingDiarizationBuffer,
    speaker_timeline: SpeakerTimeline,
    config: DiarizationSessionConfig,
}

impl DiarizationSession {
    pub fn new(embedding_model_path: &Path) -> Result<Self, EmbeddingError> {
        Self::with_profiles(embedding_model_path, Vec::new())
    }

    /// Create a session pre-seeded with saved voice profiles (name, centroid)
    /// so returning speakers are labeled by name instead of "Speaker N".
    pub fn with_profiles(
        embedding_model_path: &Path,
        profiles: Vec<(String, Vec<f32>)>,
    ) -> Result<Self, EmbeddingError> {
        Self::with_profiles_and_config(
            embedding_model_path,
            profiles,
            DiarizationSessionConfig::default(),
        )
    }

    pub fn with_profiles_and_config(
        embedding_model_path: &Path,
        profiles: Vec<(String, Vec<f32>)>,
        config: DiarizationSessionConfig,
    ) -> Result<Self, EmbeddingError> {
        let mut clusterer = SpeakerClusterer::with_config(config.clustering);
        for (name, centroid) in profiles {
            clusterer.seed_profile(&name, centroid);
        }
        Ok(Self {
            extractor: EmbeddingExtractor::new(embedding_model_path)?,
            clusterer,
            rolling_buffer: RollingDiarizationBuffer::new(
                DIARIZATION_SAMPLE_RATE,
                DEFAULT_DIARIZATION_WINDOW_SECONDS,
                DEFAULT_DIARIZATION_STRIDE_SECONDS,
            ),
            speaker_timeline: SpeakerTimeline::new(),
            config,
        })
    }

    /// (label, centroid, segment count) snapshot for persisting this
    /// recording's speakers (written to speakers.json at recording end).
    pub fn centroid_snapshot(&self) -> Vec<(String, Vec<f32>, usize)> {
        self.clusterer
            .centroids()
            .map(|(label, centroid, count)| (label.to_string(), centroid.to_vec(), count))
            .collect()
    }

    pub fn model_id(&self) -> &'static str {
        self.config.model_id
    }

    /// Assign a speaker label to a 16kHz mono speech segment.
    /// Returns None only when no label can be produced (e.g. first segment
    /// is too short). Diarization failures must never break transcription —
    /// errors are logged and degrade to the previous label or None.
    pub fn label_segment(&mut self, samples_16k: &[f32]) -> Option<String> {
        if !has_enough_samples_for_reliable_segment(
            samples_16k.len(),
            self.config.min_reliable_segment_ms,
        ) {
            return self.clusterer.last_label();
        }
        match self.extractor.compute(samples_16k) {
            Ok(embedding) => Some(self.clusterer.assign(&embedding)),
            Err(e) => {
                log::warn!(
                    "Diarization embedding failed, carrying previous label: {}",
                    e
                );
                self.clusterer.last_label()
            }
        }
    }

    /// Observe a 16kHz mono ASR chunk at its recording-relative timestamp,
    /// update the rolling diarization timeline when enough context exists,
    /// then align the ASR chunk back onto the best speaker label.
    pub fn label_segment_at(&mut self, start_time: f64, samples_16k: &[f32]) -> Option<String> {
        let duration = samples_16k.len() as f64 / DIARIZATION_SAMPLE_RATE as f64;
        let end_time = start_time + duration;

        for window in self.rolling_buffer.push_samples_at(start_time, samples_16k) {
            if !has_enough_samples_for_reliable_segment(
                window.samples.len(),
                self.config.min_reliable_segment_ms,
            ) {
                continue;
            }

            match self.extractor.compute(&window.samples) {
                Ok(embedding) => {
                    let label = self.clusterer.assign(&embedding);
                    self.speaker_timeline
                        .push_window_segment(SpeakerTimelineSegment {
                            start_time: window.start_time,
                            end_time: window.end_time,
                            speaker_ids: vec![label],
                            confidence: 0.8,
                            overlap: false,
                        });
                }
                Err(e) => {
                    log::warn!("Diarization window embedding failed: {}", e);
                }
            }
        }

        let timeline_label = self
            .speaker_timeline
            .speaker_label_for_range(start_time, end_time);

        let direct_segment_label = if has_enough_samples_for_reliable_segment(
            samples_16k.len(),
            self.config.min_reliable_segment_ms,
        ) {
            self.label_segment(samples_16k)
        } else {
            None
        };

        select_live_speaker_label(
            timeline_label,
            direct_segment_label,
            self.clusterer.last_label(),
        )
    }

    /// Assign a speaker label to an already-segmented speech range, without the
    /// live rolling window. Batch import has complete VAD segments; using the
    /// live 10s window can mix alternating speakers into one embedding.
    pub fn label_discrete_segment_at(
        &mut self,
        start_time: f64,
        samples_16k: &[f32],
    ) -> Option<String> {
        let duration = samples_16k.len() as f64 / DIARIZATION_SAMPLE_RATE as f64;
        let end_time = start_time + duration;

        if !has_enough_samples_for_reliable_segment(
            samples_16k.len(),
            self.config.min_reliable_segment_ms,
        ) {
            return self.clusterer.last_label();
        }

        match self.extractor.compute(samples_16k) {
            Ok(embedding) => {
                let label = self.clusterer.assign(&embedding);
                self.speaker_timeline
                    .push_window_segment(SpeakerTimelineSegment {
                        start_time,
                        end_time,
                        speaker_ids: vec![label.clone()],
                        confidence: 0.8,
                        overlap: false,
                    });
                Some(label)
            }
            Err(e) => {
                log::warn!(
                    "Diarization segment embedding failed, carrying previous label: {}",
                    e
                );
                self.clusterer.last_label()
            }
        }
    }

    /// Label already-segmented import audio with an offline global clustering
    /// pass. Live recording still uses the online path above.
    pub fn label_discrete_segments_offline(
        &mut self,
        segments: &[(f64, &[f32])],
    ) -> Vec<Option<String>> {
        self.speaker_timeline = SpeakerTimeline::new();

        let mut records = Vec::new();
        for (segment_index, (start_time, samples_16k)) in segments.iter().enumerate() {
            if !has_enough_samples_for_reliable_segment(
                samples_16k.len(),
                self.config.min_reliable_segment_ms,
            ) {
                continue;
            }

            match self.extractor.compute(samples_16k) {
                Ok(embedding) => {
                    let duration = samples_16k.len() as f64 / DIARIZATION_SAMPLE_RATE as f64;
                    records.push(OfflineEmbedding {
                        segment_index,
                        start_time: *start_time,
                        end_time: *start_time + duration,
                        embedding,
                    });
                }
                Err(e) => {
                    log::warn!("Diarization import embedding failed: {}", e);
                }
            }
        }

        let mut labels = vec![None; segments.len()];
        let mut clusters = offline_clusters(&records, self.config.clustering);
        clusters.sort_by(|a, b| {
            let a_start = a
                .members
                .iter()
                .map(|index| records[*index].start_time)
                .fold(f64::INFINITY, f64::min);
            let b_start = b
                .members
                .iter()
                .map(|index| records[*index].start_time)
                .fold(f64::INFINITY, f64::min);
            a_start
                .partial_cmp(&b_start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for cluster in clusters {
            let label = self.clusterer.assign(&cluster.centroid);
            for record_index in cluster.members {
                let record = &records[record_index];
                labels[record.segment_index] = Some(label.clone());
                self.speaker_timeline
                    .push_window_segment(SpeakerTimelineSegment {
                        start_time: record.start_time,
                        end_time: record.end_time,
                        speaker_ids: vec![label.clone()],
                        confidence: 0.8,
                        overlap: false,
                    });
            }
        }

        let mut last_label = None;
        for label in &mut labels {
            if label.is_some() {
                last_label = label.clone();
            } else {
                *label = last_label.clone();
            }
        }

        labels
    }

    pub fn timeline_snapshot(&self) -> Vec<SpeakerTimelineSegment> {
        self.speaker_timeline.segments().to_vec()
    }

    pub fn clusterer(&self) -> &SpeakerClusterer {
        &self.clusterer
    }

    pub fn clusterer_mut(&mut self) -> &mut SpeakerClusterer {
        &mut self.clusterer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }

    fn offline_record(segment_index: usize, embedding: Vec<f32>) -> OfflineEmbedding {
        OfflineEmbedding {
            segment_index,
            start_time: segment_index as f64,
            end_time: segment_index as f64 + 1.0,
            embedding,
        }
    }

    #[test]
    fn embedding_gate_matches_minimum_fbank_frames() {
        assert!(!has_enough_samples_for_embedding(
            MIN_SAMPLES_FOR_EMBEDDING - 1
        ));
        assert!(has_enough_samples_for_embedding(MIN_SAMPLES_FOR_EMBEDDING));
    }

    #[test]
    fn live_label_uses_warmup_label_before_timeline_exists() {
        assert_eq!(
            select_live_speaker_label(None, Some("Speaker 1".to_string()), None).as_deref(),
            Some("Speaker 1")
        );
    }

    #[test]
    fn live_label_uses_timeline_when_direct_segment_label_is_absent() {
        assert_eq!(
            select_live_speaker_label(
                Some("Speaker 2".to_string()),
                None,
                Some("Speaker 3".to_string())
            )
            .as_deref(),
            Some("Speaker 2")
        );
    }

    #[test]
    fn live_label_prefers_direct_segment_over_timeline_for_chunk_turns() {
        assert_eq!(
            select_live_speaker_label(
                Some("Speaker 2".to_string()),
                Some("Speaker 1".to_string()),
                Some("Speaker 2".to_string())
            )
            .as_deref(),
            Some("Speaker 1")
        );
    }

    #[test]
    fn live_label_computes_direct_label_when_timeline_misses_after_startup() {
        assert!(should_compute_direct_segment_label(
            MIN_SAMPLES_FOR_EMBEDDING
        ));
    }

    #[test]
    fn offline_clustering_uses_global_similarity() {
        let records = vec![
            offline_record(0, unit(vec![1.0, 0.0, 0.0])),
            offline_record(1, unit(vec![0.0, 1.0, 0.0])),
            offline_record(2, unit(vec![0.95, 0.05, 0.0])),
        ];
        let clusters = offline_clusters(
            &records,
            SpeakerClusteringConfig {
                cluster_similarity_threshold: 0.9,
                profile_match_threshold: 0.9,
                max_anonymous_speakers: 8,
            },
        );

        assert_eq!(clusters.len(), 2);
        assert!(clusters.iter().any(|cluster| cluster.members.len() == 2));
    }

    #[test]
    fn offline_clustering_respects_max_speakers() {
        let records = vec![
            offline_record(0, unit(vec![1.0, 0.0, 0.0])),
            offline_record(1, unit(vec![0.0, 1.0, 0.0])),
            offline_record(2, unit(vec![0.0, 0.0, 1.0])),
        ];
        let clusters = offline_clusters(
            &records,
            SpeakerClusteringConfig {
                cluster_similarity_threshold: 0.99,
                profile_match_threshold: 0.99,
                max_anonymous_speakers: 2,
            },
        );

        assert_eq!(clusters.len(), 2);
    }
}
