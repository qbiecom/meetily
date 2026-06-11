'use client';

// Speaker label chip for diarized transcripts. Color is derived
// deterministically from the label so a speaker keeps the same color
// across the live view, saved meetings, and app restarts.

import { Pencil } from 'lucide-react';

const SPEAKER_PALETTE = [
  { bg: 'bg-blue-50', text: 'text-blue-700', ring: 'ring-blue-200', avatar: 'bg-gradient-to-br from-blue-400 to-blue-600' },
  { bg: 'bg-emerald-50', text: 'text-emerald-700', ring: 'ring-emerald-200', avatar: 'bg-gradient-to-br from-emerald-400 to-emerald-600' },
  { bg: 'bg-purple-50', text: 'text-purple-700', ring: 'ring-purple-200', avatar: 'bg-gradient-to-br from-purple-400 to-purple-600' },
  { bg: 'bg-amber-50', text: 'text-amber-700', ring: 'ring-amber-200', avatar: 'bg-gradient-to-br from-amber-400 to-amber-600' },
  { bg: 'bg-rose-50', text: 'text-rose-700', ring: 'ring-rose-200', avatar: 'bg-gradient-to-br from-rose-400 to-rose-600' },
  { bg: 'bg-cyan-50', text: 'text-cyan-700', ring: 'ring-cyan-200', avatar: 'bg-gradient-to-br from-cyan-400 to-cyan-600' },
  { bg: 'bg-indigo-50', text: 'text-indigo-700', ring: 'ring-indigo-200', avatar: 'bg-gradient-to-br from-indigo-400 to-indigo-600' },
  { bg: 'bg-orange-50', text: 'text-orange-700', ring: 'ring-orange-200', avatar: 'bg-gradient-to-br from-orange-400 to-orange-600' },
];

export function speakerColor(label: string) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) | 0;
  }
  return SPEAKER_PALETTE[Math.abs(hash) % SPEAKER_PALETTE.length];
}

// Initials for the avatar: "Speaker 1" → "S1", "Alice Wong" → "AW", "Bob" → "B".
function speakerInitials(label: string) {
  const trimmed = label.trim();
  const speakerMatch = trimmed.match(/^speaker\s*(\d+)$/i);
  if (speakerMatch) return `S${speakerMatch[1]}`;
  const words = trimmed.split(/\s+/).filter(Boolean);
  if (words.length === 0) return '?';
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
}

interface SpeakerChipProps {
  label: string;
  onClick?: () => void;
  className?: string;
}

export function SpeakerChip({ label, onClick, className = '' }: SpeakerChipProps) {
  const color = speakerColor(label);
  const initials = speakerInitials(label);
  return (
    <span
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      className={`group inline-flex items-center gap-1.5 pl-0.5 pr-2 py-0.5 rounded-full text-xs font-medium ring-1 ring-inset transition-all duration-150 ${color.bg} ${color.text} ${color.ring} ${
        onClick
          ? 'cursor-pointer hover:shadow-sm hover:-translate-y-px focus:outline-none focus-visible:ring-2'
          : ''
      } ${className}`}
      title={onClick ? 'Click to rename speaker' : undefined}
    >
      <span
        className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[9px] font-semibold leading-none text-white shadow-inner ${color.avatar}`}
      >
        {initials}
      </span>
      <span className="truncate">{label}</span>
      {onClick && (
        <Pencil className="h-2.5 w-2.5 shrink-0 opacity-0 transition-opacity duration-150 group-hover:opacity-60" />
      )}
    </span>
  );
}
