import { formatDistanceToNow } from 'date-fns';
import type { Goal, ScheduledPrompt } from '@/api/types';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { Target, CalendarClock } from 'lucide-react';

interface RecentActivityProps {
  goals: Goal[];
  schedules: ScheduledPrompt[];
}

export function RecentActivity({ goals, schedules }: RecentActivityProps) {
  const recentGoals = [...goals]
    .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
    .slice(0, 5);

  const recentSchedules = [...schedules]
    .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
    .slice(0, 5);

  return (
    <div className="obsidian p-5">
      <div className="flex items-center gap-3 mb-5">
        <span className="label-text" style={{ color: 'var(--color-primary)' }}>Recent Activity</span>
        <div className="brass-line flex-1" />
      </div>
      <div className="space-y-3">
        {recentGoals.map((goal) => (
          <div
            key={goal.id}
            className="flex items-center gap-3 text-sm py-1.5 px-2 rounded-md transition-colors"
            style={{ borderBottom: '1px solid var(--color-border)' }}
          >
            <Target className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--color-primary)' }} />
            <span className="flex-1 truncate readout" style={{ color: 'var(--color-text)', fontSize: '0.8rem' }}>
              {goal.goal}
            </span>
            <StatusBadge status={goal.status} />
            {goal.created_at && (
              <span className="label-text" style={{ minWidth: '80px', textAlign: 'right' }}>
                {formatDistanceToNow(new Date(goal.created_at), { addSuffix: true })}
              </span>
            )}
          </div>
        ))}
        {recentSchedules.map((schedule) => (
          <div
            key={schedule.id}
            className="flex items-center gap-3 text-sm py-1.5 px-2 rounded-md transition-colors"
            style={{ borderBottom: '1px solid var(--color-border)' }}
          >
            <CalendarClock className="h-3.5 w-3.5 shrink-0" style={{ color: 'var(--color-accent-amber)' }} />
            <span className="flex-1 truncate readout" style={{ color: 'var(--color-text)', fontSize: '0.8rem' }}>
              {schedule.prompt}
            </span>
            <StatusBadge status={schedule.status} />
          </div>
        ))}
        {recentGoals.length === 0 && recentSchedules.length === 0 && (
          <p className="text-sm py-4" style={{ color: 'var(--color-text-dim)' }}>No recent activity</p>
        )}
      </div>
    </div>
  );
}