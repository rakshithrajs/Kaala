import { useQuery } from '@tanstack/react-query';
import { goalsApi } from '@/api/goals';
import { schedulesApi } from '@/api/schedules';
import { AnimatedPage } from '@/components/shared/AnimatedPage';
import { StatsCard } from '@/components/dashboard/StatsCard';
import { RecentActivity } from '@/components/dashboard/RecentActivity';
import { AgentFlowDiagram } from '@/components/dashboard/AgentFlowDiagram';
import { LoadingState } from '@/components/shared/LoadingState';
import { Target, CalendarClock, CheckCircle2, Clock } from 'lucide-react';

export function DashboardPage() {
  const { data: goals, isLoading: goalsLoading } = useQuery({
    queryKey: ['goals'],
    queryFn: () => goalsApi.list(),
  });

  const { data: schedules, isLoading: schedulesLoading } = useQuery({
    queryKey: ['schedules'],
    queryFn: () => schedulesApi.list(),
  });

  if (goalsLoading || schedulesLoading) return <LoadingState rows={3} />;

  const allGoals = goals ?? [];
  const allSchedules = schedules ?? [];
  const activeGoals = allGoals.filter((g) => g.status === 'pending' || g.status === 'in_progress').length;
  const pendingSchedules = allSchedules.filter((s) => s.status === 'pending').length;
  const completedToday = allGoals.filter((g) => {
    if (g.status !== 'completed' || !g.completed_at) return false;
    const completed = new Date(g.completed_at);
    const today = new Date();
    return completed.toDateString() === today.toDateString();
  }).length;

  return (
    <AnimatedPage>
      <div className="space-y-6">
        {/* Section header */}
        <div className="flex items-center gap-4 mb-2">
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.75rem', color: 'var(--color-text)' }}>
            Observatory
          </h2>
          <div className="brass-line flex-1" />
          <span className="label-text" style={{ color: 'var(--color-primary-dim)' }}>
            {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
          </span>
        </div>

        {/* Stats cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatsCard icon={Target} label="Total Goals" value={allGoals.length} color="var(--color-primary)" />
          <StatsCard icon={Clock} label="Active Goals" value={activeGoals} color="var(--color-accent-blue)" />
          <StatsCard icon={CalendarClock} label="Pending Schedules" value={pendingSchedules} color="var(--color-accent-amber)" />
          <StatsCard icon={CheckCircle2} label="Completed Today" value={completedToday} color="var(--color-accent-green)" />
        </div>

        {/* Activity + Pipeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <RecentActivity goals={allGoals} schedules={allSchedules} />
          <AgentFlowDiagram />
        </div>
      </div>
    </AnimatedPage>
  );
}