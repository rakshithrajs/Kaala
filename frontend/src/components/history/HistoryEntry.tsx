import { formatDistanceToNow } from 'date-fns';
import { User, Bot } from 'lucide-react';
import { AGENT_COLORS } from '@/api/types';
import type { HistoryEntry } from '@/api/types';

interface HistoryEntryProps {
  entry: HistoryEntry;
}

export function HistoryEntry({ entry }: HistoryEntryProps) {
  const isUser = entry.role === 'user';
  const agentColor = isUser ? 'var(--color-primary)' : (AGENT_COLORS[entry.agent?.toLowerCase()] ?? 'var(--color-text-muted)');

  return (
    <div className="obsidian p-3 flex items-start gap-3">
      <div
        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md"
        style={{
          backgroundColor: `color-mix(in srgb, ${agentColor} 15%, transparent)`,
          border: `1px solid color-mix(in srgb, ${agentColor} 25%, transparent)`,
        }}
      >
        {isUser ? (
          <User className="h-3.5 w-3.5" style={{ color: agentColor }} />
        ) : (
          <Bot className="h-3.5 w-3.5" style={{ color: agentColor }} />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className="text-[10px] font-semibold uppercase tracking-wider"
            style={{ color: agentColor, fontFamily: 'var(--font-label)' }}
          >
            {entry.agent}
          </span>
          {entry.timestamp && (
            <span className="label-text">
              {formatDistanceToNow(new Date(entry.timestamp), { addSuffix: true })}
            </span>
          )}
        </div>
        <p className="text-sm readout line-clamp-3" style={{ color: 'var(--color-text)' }}>{entry.content}</p>
      </div>
    </div>
  );
}