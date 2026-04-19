import { useState } from 'react';
import { useGoals, useUpdateGoal, useDeleteGoal } from '@/hooks/useGoals';
import { AnimatedPage } from '@/components/shared/AnimatedPage';
import { GoalStatusTabs } from '@/components/goals/GoalStatusTabs';
import { GoalCard } from '@/components/goals/GoalCard';
import { ConfirmDialog, useConfirmDialog } from '@/components/shared/ConfirmDialog';
import { LoadingState } from '@/components/shared/LoadingState';
import { EmptyState } from '@/components/shared/EmptyState';
import { Target } from 'lucide-react';
import type { GoalStatus } from '@/api/types';

const TABS: { label: string; value: GoalStatus | undefined }[] = [
  { label: 'All', value: undefined },
  { label: 'Pending', value: 'pending' },
  { label: 'In Progress', value: 'in_progress' },
  { label: 'Completed', value: 'completed' },
  { label: 'Cancelled', value: 'cancelled' },
];

export function GoalsPage() {
  const [activeTab, setActiveTab] = useState<string | undefined>(undefined);
  const { data: goals, isLoading } = useGoals(activeTab);
  const updateGoal = useUpdateGoal();
  const deleteGoal = useDeleteGoal();
  const { dialogState, confirm, cancel } = useConfirmDialog();

  if (isLoading) return <LoadingState rows={4} />;
  const allGoals = goals ?? [];

  return (
    <AnimatedPage>
      <div className="flex items-center gap-3 mb-4">
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: 'var(--color-text)' }}>
          Goals
        </h2>
        <div className="brass-line flex-1" />
      </div>
      <GoalStatusTabs tabs={TABS} active={activeTab} onChange={setActiveTab} />
      <div className="mt-4 space-y-3">
        {allGoals.length === 0 ? (
          <EmptyState icon={Target} title="No goals found" description="Start a conversation to create goals" />
        ) : (
          allGoals.map((goal) => (
            <GoalCard
              key={goal.id}
              goal={goal}
              onComplete={(id) => updateGoal.mutate({ id, data: { status: 'completed' } })}
              onCancel={(id) => updateGoal.mutate({ id, data: { status: 'cancelled' } })}
              onDelete={(id) =>
                confirm({ title: 'Delete Goal', message: 'Are you sure you want to delete this goal?' }).then(() =>
                  deleteGoal.mutate(id)
                )
              }
            />
          ))
        )}
      </div>
      <ConfirmDialog {...dialogState} onCancel={cancel} confirmLabel="Delete" />
    </AnimatedPage>
  );
}