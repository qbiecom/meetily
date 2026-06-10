'use client';

// Speaker label chip for diarized transcripts. Color is derived
// deterministically from the label so a speaker keeps the same color
// across the live view, saved meetings, and app restarts.

const SPEAKER_PALETTE = [
  { bg: 'bg-blue-100', text: 'text-blue-700', dot: 'bg-blue-500' },
  { bg: 'bg-emerald-100', text: 'text-emerald-700', dot: 'bg-emerald-500' },
  { bg: 'bg-purple-100', text: 'text-purple-700', dot: 'bg-purple-500' },
  { bg: 'bg-amber-100', text: 'text-amber-700', dot: 'bg-amber-500' },
  { bg: 'bg-rose-100', text: 'text-rose-700', dot: 'bg-rose-500' },
  { bg: 'bg-cyan-100', text: 'text-cyan-700', dot: 'bg-cyan-500' },
  { bg: 'bg-indigo-100', text: 'text-indigo-700', dot: 'bg-indigo-500' },
  { bg: 'bg-orange-100', text: 'text-orange-700', dot: 'bg-orange-500' },
];

export function speakerColor(label: string) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) | 0;
  }
  return SPEAKER_PALETTE[Math.abs(hash) % SPEAKER_PALETTE.length];
}

interface SpeakerChipProps {
  label: string;
  onClick?: () => void;
}

export function SpeakerChip({ label, onClick }: SpeakerChipProps) {
  const color = speakerColor(label);
  return (
    <span
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${color.bg} ${color.text} ${
        onClick ? 'cursor-pointer hover:opacity-80' : ''
      }`}
      title={onClick ? 'Click to rename speaker' : undefined}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${color.dot}`} />
      {label}
    </span>
  );
}
