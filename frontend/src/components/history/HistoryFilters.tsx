interface HistoryFiltersProps {
  agent: string | undefined;
  onAgentChange: (agent: string | undefined) => void;
  limit: number;
  onLimitChange: (limit: number) => void;
}

const AGENTS = ['All', 'user', 'niyati', 'iccha', 'karya', 'karma'];
const LIMITS = [20, 50, 100, 200];

export function HistoryFilters({ agent, onAgentChange, limit, onLimitChange }: HistoryFiltersProps) {
  return (
    <div className="flex items-center gap-4">
      <select
        value={agent ?? 'All'}
        onChange={(e) => onAgentChange(e.target.value === 'All' ? undefined : e.target.value)}
        className="rounded-md px-3 py-1.5 text-sm readout"
        style={{
          backgroundColor: 'var(--color-surface)',
          color: 'var(--color-text)',
          border: '1px solid var(--color-border)',
          outline: 'none',
          fontFamily: 'var(--font-label)',
        }}
      >
        {AGENTS.map((a) => (
          <option key={a} value={a}>{a === 'All' ? 'All Agents' : a.charAt(0).toUpperCase() + a.slice(1)}</option>
        ))}
      </select>
      <select
        value={limit}
        onChange={(e) => onLimitChange(Number(e.target.value))}
        className="rounded-md px-3 py-1.5 text-sm readout"
        style={{
          backgroundColor: 'var(--color-surface)',
          color: 'var(--color-text)',
          border: '1px solid var(--color-border)',
          outline: 'none',
          fontFamily: 'var(--font-label)',
        }}
      >
        {LIMITS.map((l) => (
          <option key={l} value={l}>{l} entries</option>
        ))}
      </select>
    </div>
  );
}