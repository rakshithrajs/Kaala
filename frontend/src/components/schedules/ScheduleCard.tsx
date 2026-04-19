import { formatDistanceToNow } from 'date-fns';
import { XCircle, CalendarClock } from 'lucide-react';
import { StatusBadge } from '@/components/shared/StatusBadge';
import type { ScheduledPrompt } from '@/api/types';

interface ScheduleCardProps {
  schedule: ScheduledPrompt;
  onCancel: (id: number) => void;
}

export function ScheduleCard({ schedule, onCancel }: ScheduleCardProps) {
  return (
    <div className="obsidian obsidian-hover p-4 transition-all duration-200" style={{ borderLeft: '2px solid var(--color-accent-amber)' }}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <CalendarClock className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--color-accent-amber)' }} />
            <p className="text-sm font-medium truncate readout" style={{ color: 'var(--color-text)' }}>{schedule.prompt}</p>
          </div>
          <div className="flex items-center gap-3 mt-2 ml-5">
            <StatusBadge status={schedule.status} />
            <span
              className="inline-flex items-center rounded px-2 py-0.5 readout"
              style={{
                backgroundColor: 'rgba(251,191,36,0.1)',
                color: 'var(--color-accent-amber)',
                border: '1px solid rgba(251,191,36,0.2)',
                fontSize: '0.65rem',
                fontFamily: 'var(--font-label)',
                fontWeight: 500,
                letterSpacing: '0.05em',
                textTransform: 'uppercase',
              }}
            >
              {schedule.prompt_type.replace('_', ' ')}
            </span>
            {schedule.goal_id && (
              <span className="label-text">Goal #{schedule.goal_id}</span>
            )}
          </div>
          <div className="flex items-center gap-3 mt-1 ml-5">
            {schedule.scheduled_for && (
              <span className="label-text">
                Scheduled: {formatDistanceToNow(new Date(schedule.scheduled_for), { addSuffix: true })}
              </span>
            )}
            {schedule.executed_at && (
              <span className="label-text" style={{ color: 'var(--color-accent-green)' }}>
                Executed: {formatDistanceToNow(new Date(schedule.executed_at), { addSuffix: true })}
              </span>
            )}
          </div>
        </div>
        {schedule.status === 'pending' && (
          <button
            onClick={() => onCancel(schedule.id)}
            className="rounded-md p-1.5 transition-colors"
            style={{ color: 'var(--color-text-muted)' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--color-accent-red)'; e.currentTarget.style.background = 'rgba(248,113,113,0.1)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--color-text-muted)'; e.currentTarget.style.background = 'transparent'; }}
            title="Cancel"
          >
            <XCircle className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}