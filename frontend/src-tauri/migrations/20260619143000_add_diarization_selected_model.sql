-- Selected speaker embedding model for local speaker identification.
ALTER TABLE diarization_settings
ADD COLUMN selected_model_id TEXT NOT NULL DEFAULT '3dspeaker-eres2net-en';
