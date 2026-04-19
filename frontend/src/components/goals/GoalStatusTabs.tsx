import { cn } from '@/lib/utils';

interface Tab {
  label: string;
  value: string | undefined;
}

interface GoalStatusTabsProps {
  tabs: Tab[];
  active: string | undefined;
  onChange: (value: string | undefined) => void;
}

export function GoalStatusTabs({ tabs, active, onChange }: GoalStatusTabsProps) {
  return (
    <div
      className="flex gap-1 rounded-md p-1"
      style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
    >
      {tabs.map((tab) => (
        <button
          key={tab.value ?? 'all'}
          onClick={() => onChange(tab.value)}
          className={cn('rounded-md px-3 py-1.5 text-xs font-medium transition-all')}
          style={{
            fontFamily: 'var(--font-label)',
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
            background: active === tab.value ? 'rgba(200,164,78,0.15)' : 'transparent',
            color: active === tab.value ? 'var(--color-primary)' : 'var(--color-text-muted)',
            border: active === tab.value ? '1px solid rgba(200,164,78,0.3)' : '1px solid transparent',
          }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}