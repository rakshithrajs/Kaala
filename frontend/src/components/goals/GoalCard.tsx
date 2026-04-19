import { formatDistanceToNow } from 'date-fns';
import { CheckCircle2, XCircle, Trash2, Target } from 'lucide-react';
import { StatusBadge } from '@/components/shared/StatusBadge';
import type { Goal } from '@/api/types';

interface GoalCardProps {
  goal: Goal;
  onComplete: (id: number) => void;
  onCancel: (id: number) => void;
  onDelete: (id: number) => void;
}

export function GoalCard({ goal, onComplete, onCancel, onDelete }: GoalCardProps) {
  return (
    <div className="obsidian obsidian-hover p-4 transition-all duration-200" style={{ borderLeft: '2px solid var(--color-primary-dim)' }}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Target className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--color-primary)' }} />
            <p className="text-sm font-medium truncate readout" style={{ color: 'var(--color-text)' }}>{goal.goal}</p>
          </div>
          {goal.details && (
            <p className="text-xs ml-5 line-clamp-2 readout" style={{ color: 'var(--color-text-muted)' }}>{goal.details}</p>
          )}
          <div className="flex items-center gap-3 mt-2 ml-5">
            <StatusBadge status={goal.status} />
            {goal.created_at && (
              <span className="label-text">
                {formatDistanceToNow(new Date(goal.created_at), { addSuffix: true })}
              </span>
            )}
            {goal.completed_at && goal.status === 'completed' && (
              <span className="label-text" style={{ color: 'var(--color-accent-green)' }}>
                completed {formatDistanceToNow(new Date(goal.completed_at), { addSuffix: true })}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {(goal.status === 'pending' || goal.status === 'in_progress') && (
            <>
              <button
                onClick={() => onComplete(goal.id)}
                className="rounded-md p-1.5 transition-colors"
                style={{ color: 'var(--color-accent-green)' }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(74,222,128,0.1)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                title="Complete"
              >
                <CheckCircle2 className="h-4 w-4" />
              </button>
              <button
                onClick={() => onCancel(goal.id)}
                className="rounded-md p-1.5 transition-colors"
                style={{ color: 'var(--color-text-muted)' }}
                onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--color-accent-red)'; e.currentTarget.style.background = 'rgba(248,113,113,0.1)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--color-text-muted)'; e.currentTarget.style.background = 'transparent'; }}
                title="Cancel"
              >
                <XCircle className="h-4 w-4" />
              </button>
            </>
          )}
          {(goal.status === 'completed' || goal.status === 'cancelled') && (
            <button
              onClick={() => onDelete(goal.id)}
              className="rounded-md p-1.5 transition-colors"
              style={{ color: 'var(--color-text-muted)' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--color-accent-red)'; e.currentTarget.style.background = 'rgba(248,113,113,0.1)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--color-text-muted)'; e.currentTarget.style.background = 'transparent'; }}
              title="Delete"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}