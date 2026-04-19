import type { Goal } from '@/api/types';

interface GoalCompletionChartProps {
  goals: Goal[];
}

export function GoalCompletionChart({ goals }: GoalCompletionChartProps) {
  const statusCounts = {
    pending: goals.filter((g) => g.status === 'pending').length,
    in_progress: goals.filter((g) => g.status === 'in_progress').length,
    completed: goals.filter((g) => g.status === 'completed').length,
    cancelled: goals.filter((g) => g.status === 'cancelled').length,
  };

  const total = goals.length;
  const colors: Record<string, string> = {
    pending: '#fbbf24',
    in_progress: '#60a5fa',
    completed: '#4ade80',
    cancelled: '#6e7490',
  };

  let cumulativePercent = 0;
  const segments = Object.entries(statusCounts)
    .filter(([, count]) => count > 0)
    .map(([status, count]) => {
      const percent = total > 0 ? (count / total) * 100 : 0;
      const segment = {
        status,
        count,
        percent,
        startAngle: cumulativePercent * 3.6,
        color: colors[status],
      };
      cumulativePercent += percent;
      return segment;
    });

  return (
    <div className="obsidian p-5">
      <div className="flex items-center gap-3 mb-5">
        <span className="label-text" style={{ color: 'var(--color-primary)' }}>Goal Completion</span>
        <div className="brass-line flex-1" />
      </div>
      {total === 0 ? (
        <p style={{ color: 'var(--color-text-dim)' }} className="text-sm readout">No goals yet</p>
      ) : (
        <div className="flex items-center gap-8">
          <div className="relative w-36 h-36 shrink-0">
            <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
              {segments.map((seg) => (
                <circle
                  key={seg.status}
                  r="15.915"
                  cx="18"
                  cy="18"
                  fill="transparent"
                  stroke={seg.color}
                  strokeWidth="3.5"
                  style={{
                    strokeDasharray: `${seg.percent} ${100 - seg.percent}`,
                    strokeDashoffset: -(segments.slice(0, segments.indexOf(seg)).reduce((sum, s) => sum + s.percent, 0)),
                  }}
                />
              ))}
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl readout" style={{ color: 'var(--color-text)', fontFamily: 'var(--font-body)', fontWeight: 700 }}>{total}</span>
            </div>
          </div>
          <div className="space-y-2.5">
            {Object.entries(statusCounts).map(([status, count]) => (
              <div key={status} className="flex items-center gap-2 text-sm">
                <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: colors[status] }} />
                <span className="label-text capitalize" style={{ minWidth: '70px' }}>{status.replace('_', ' ')}</span>
                <span className="readout" style={{ color: 'var(--color-text)', fontFamily: 'var(--font-body)', fontWeight: 600 }}>{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}