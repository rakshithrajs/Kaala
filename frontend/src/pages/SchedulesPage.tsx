import { useState } from 'react';
import { useSchedules, useCancelSchedule } from '@/hooks/useSchedules';
import { AnimatedPage } from '@/components/shared/AnimatedPage';
import { ScheduleStatusTabs } from '@/components/schedules/ScheduleStatusTabs';
import { ScheduleCard } from '@/components/schedules/ScheduleCard';
import { LoadingState } from '@/components/shared/LoadingState';
import { EmptyState } from '@/components/shared/EmptyState';
import { CalendarClock } from 'lucide-react';
import type { ScheduleStatus } from '@/api/types';

const TABS: { label: string; value: ScheduleStatus | undefined }[] = [
  { label: 'All', value: undefined },
  { label: 'Pending', value: 'pending' },
  { label: 'Executed', value: 'executed' },
  { label: 'Failed', value: 'failed' },
  { label: 'Cancelled', value: 'cancelled' },
];

export function SchedulesPage() {
  const [activeTab, setActiveTab] = useState<string | undefined>(undefined);
  const { data: schedules, isLoading } = useSchedules(activeTab);
  const cancelSchedule = useCancelSchedule();

  if (isLoading) return <LoadingState rows={4} />;
  const allSchedules = schedules ?? [];

  return (
    <AnimatedPage>
      <div className="flex items-center gap-3 mb-4">
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: 'var(--color-text)' }}>
          Ephemeris
        </h2>
        <div className="brass-line flex-1" />
      </div>
      <ScheduleStatusTabs tabs={TABS} active={activeTab} onChange={setActiveTab} />
      <div className="mt-4 space-y-3">
        {allSchedules.length === 0 ? (
          <EmptyState icon={CalendarClock} title="No schedules found" description="Goals with future follow-ups will appear here" />
        ) : (
          allSchedules.map((schedule) => (
            <ScheduleCard
              key={schedule.id}
              schedule={schedule}
              onCancel={(id) => cancelSchedule.mutate(id)}
            />
          ))
        )}
      </div>
    </AnimatedPage>
  );
}