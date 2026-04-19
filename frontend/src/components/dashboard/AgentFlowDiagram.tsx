const agents = [
  { name: 'Niyati', role: 'Orchestrator', color: 'var(--color-agent-niyati)', glow: 'agent-glow-niyati' },
  { name: 'Iccha', role: 'Goal Extractor', color: 'var(--color-agent-iccha)', glow: 'agent-glow-iccha' },
  { name: 'Karya', role: 'Planner', color: 'var(--color-agent-karya)', glow: 'agent-glow-karya' },
  { name: 'Karma', role: 'Executor', color: 'var(--color-agent-karma)', glow: 'agent-glow-karma' },
];

export function AgentFlowDiagram() {
  return (
    <div className="obsidian p-5">
      <div className="flex items-center gap-3 mb-5">
        <span className="label-text" style={{ color: 'var(--color-primary)' }}>Agent Pipeline</span>
        <div className="brass-line flex-1" />
      </div>
      <div className="flex flex-col items-center gap-2 py-2">
        {/* User Input — top node */}
        <div
          className="px-5 py-2 rounded-md text-sm readout"
          style={{
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text-muted)',
          }}
        >
          User Input
        </div>

        {/* Connector */}
        <svg width="2" height="20" className="opacity-40">
          <line x1="1" y1="0" x2="1" y2="20" stroke="var(--color-primary-dim)" strokeWidth="1" strokeDasharray="3 3" />
        </svg>

        {/* Niyati — central router */}
        <div
          className={`obsidian px-6 py-3 text-center transition-all ${agents[0].glow}`}
          style={{ borderColor: agents[0].color }}
        >
          <p className="font-semibold text-sm" style={{ color: agents[0].color, fontFamily: 'var(--font-display)' }}>
            {agents[0].name}
          </p>
          <p className="label-text mt-0.5">{agents[0].role}</p>
        </div>

        {/* Branching connector */}
        <svg width="200" height="28" viewBox="0 0 200 28" className="opacity-40">
          <line x1="100" y1="0" x2="100" y2="10" stroke="var(--color-primary-dim)" strokeWidth="1" strokeDasharray="3 3" />
          <line x1="35" y1="10" x2="165" y2="10" stroke="var(--color-primary-dim)" strokeWidth="1" strokeDasharray="3 3" />
          <line x1="35" y1="10" x2="35" y2="28" stroke="var(--color-agent-iccha)" strokeWidth="1" strokeDasharray="3 3" />
          <line x1="100" y1="10" x2="100" y2="28" stroke="var(--color-agent-karya)" strokeWidth="1" strokeDasharray="3 3" />
          <line x1="165" y1="10" x2="165" y2="28" stroke="var(--color-agent-karma)" strokeWidth="1" strokeDasharray="3 3" />
          {/* Small dots at intersections */}
          <circle cx="35" cy="10" r="2" fill="var(--color-agent-iccha)" opacity="0.6" />
          <circle cx="100" cy="10" r="2" fill="var(--color-agent-karya)" opacity="0.6" />
          <circle cx="165" cy="10" r="2" fill="var(--color-agent-karma)" opacity="0.6" />
        </svg>

        {/* Branch: Iccha, Karya, Karma */}
        <div className="grid grid-cols-3 gap-3 w-full">
          {agents.slice(1).map((agent) => (
            <div
              key={agent.name}
              className={`obsidian obsidian-hover px-3 py-3 text-center transition-all ${agent.glow}`}
              style={{ borderColor: agent.color }}
            >
              <p className="font-semibold text-sm" style={{ color: agent.color, fontFamily: 'var(--font-display)' }}>
                {agent.name}
              </p>
              <p className="label-text mt-0.5">{agent.role}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}