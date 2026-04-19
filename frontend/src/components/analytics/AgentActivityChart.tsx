import { AGENT_COLORS } from '@/api/types';

const agents = [
  { name: 'Niyati', role: 'Orchestrator' },
  { name: 'Iccha', role: 'Goal Extractor' },
  { name: 'Karya', role: 'Planner' },
  { name: 'Karma', role: 'Executor' },
];

export function AgentActivityChart() {
  return (
    <div className="obsidian p-5">
      <div className="flex items-center gap-3 mb-5">
        <span className="label-text" style={{ color: 'var(--color-primary)' }}>Agent Overview</span>
        <div className="brass-line flex-1" />
      </div>
      <div className="space-y-4">
        {agents.map((agent) => {
          const colorKey = agent.name.toLowerCase();
          const color = AGENT_COLORS[colorKey] ?? 'var(--color-text-muted)';
          return (
            <div key={agent.name} className="flex items-center gap-3">
              <div
                className="h-10 w-10 rounded-md flex items-center justify-center text-sm"
                style={{
                  backgroundColor: `color-mix(in srgb, ${color} 15%, transparent)`,
                  border: `1px solid color-mix(in srgb, ${color} 25%, transparent)`,
                  color,
                  fontFamily: 'var(--font-display)',
                  fontWeight: 700,
                }}
              >
                {agent.name[0]}
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium" style={{ color, fontFamily: 'var(--font-display)' }}>{agent.name}</p>
                <p className="label-text">{agent.role}</p>
              </div>
              <div
                className="h-1 rounded-sm"
                style={{ backgroundColor: `color-mix(in srgb, ${color} 30%, transparent)`, width: '60%', minWidth: '2rem' }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}