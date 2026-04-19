import { useQuery } from '@tanstack/react-query';
import { goalsApi } from '@/api/goals';
import { AnimatedPage } from '@/components/shared/AnimatedPage';
import { LoadingState } from '@/components/shared/LoadingState';
import { GoalCompletionChart } from '@/components/analytics/GoalCompletionChart';
import { AgentActivityChart } from '@/components/analytics/AgentActivityChart';

export function AnalyticsPage() {
  const { data: goals, isLoading: goalsLoading } = useQuery({
    queryKey: ['goals'],
    queryFn: () => goalsApi.list(),
  });

  if (goalsLoading) return <LoadingState rows={3} />;

  return (
    <AnimatedPage>
      <div className="flex items-center gap-3 mb-4">
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: 'var(--color-text)' }}>
          Observations
        </h2>
        <div className="brass-line flex-1" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GoalCompletionChart goals={goals ?? []} />
        <AgentActivityChart />
      </div>
    </AnimatedPage>
  );
}