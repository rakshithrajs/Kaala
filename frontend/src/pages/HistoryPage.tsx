import { useState } from 'react';
import { useHistory } from '@/hooks/useHistory';
import { AnimatedPage } from '@/components/shared/AnimatedPage';
import { HistoryFilters } from '@/components/history/HistoryFilters';
import { HistoryEntry } from '@/components/history/HistoryEntry';
import { LoadingState } from '@/components/shared/LoadingState';
import { EmptyState } from '@/components/shared/EmptyState';
import { History } from 'lucide-react';

export function HistoryPage() {
  const [agent, setAgent] = useState<string | undefined>(undefined);
  const [limit, setLimit] = useState(50);
  const { data: entries, isLoading } = useHistory({ agent, limit });

  if (isLoading) return <LoadingState rows={5} />;

  return (
    <AnimatedPage>
      <div className="flex items-center gap-3 mb-4">
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: 'var(--color-text)' }}>
          Chronicle
        </h2>
        <div className="brass-line flex-1" />
      </div>
      <HistoryFilters agent={agent} onAgentChange={setAgent} limit={limit} onLimitChange={setLimit} />
      <div className="mt-4 space-y-2">
        {(!entries || entries.length === 0) ? (
          <EmptyState icon={History} title="No history found" description="Conversation history will appear here" />
        ) : (
          entries.map((entry, i) => <HistoryEntry key={i} entry={entry} />)
        )}
      </div>
    </AnimatedPage>
  );
}