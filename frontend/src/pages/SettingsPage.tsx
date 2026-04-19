import { AnimatedPage } from '@/components/shared/AnimatedPage';
import { useWsStore } from '@/stores/ws-store';
import { Compass, Wifi, WifiOff } from 'lucide-react';

export function SettingsPage() {
  const connected = useWsStore((s) => s.connected);

  return (
    <AnimatedPage>
      <div className="flex items-center gap-3 mb-4">
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: 'var(--color-text)' }}>
          Instrumentation
        </h2>
        <div className="brass-line flex-1" />
      </div>

      <div className="max-w-2xl space-y-6">
        {/* Connection Status */}
        <div className="obsidian p-6">
          <div className="flex items-center gap-3 mb-4">
            {connected ? (
              <Wifi className="h-4.5 w-4.5" style={{ color: 'var(--color-accent-green)' }} />
            ) : (
              <WifiOff className="h-4.5 w-4.5" style={{ color: 'var(--color-accent-red)' }} />
            )}
            <span className="label-text" style={{ color: 'var(--color-primary)' }}>Connection Status</span>
          </div>
          <p className="text-sm readout" style={{ color: 'var(--color-text-muted)' }}>
            WebSocket: <span style={{ color: connected ? 'var(--color-accent-green)' : 'var(--color-accent-red)' }}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </p>
        </div>

        {/* Display */}
        <div className="obsidian p-6">
          <div className="flex items-center gap-3 mb-4">
            <Compass className="h-4.5 w-4.5" style={{ color: 'var(--color-primary)' }} />
            <span className="label-text" style={{ color: 'var(--color-primary)' }}>Display</span>
          </div>
          <p className="text-sm readout" style={{ color: 'var(--color-text-muted)' }}>
            Dark mode is always on — the observatory operates in the dark.
          </p>
        </div>

        {/* About */}
        <div className="obsidian p-6">
          <div className="flex items-center gap-3 mb-4">
            <Compass className="h-4.5 w-4.5" style={{ color: 'var(--color-accent-amber)' }} />
            <span className="label-text" style={{ color: 'var(--color-primary)' }}>About Kaala</span>
          </div>
          <p className="text-sm readout mb-3" style={{ color: 'var(--color-text-muted)' }}>
            Kaala is a multi-agent goal-tracking personal assistant.
          </p>
          <div className="space-y-2.5 ml-1">
            {[
              { name: 'Niyati', role: 'Orchestrator & Router', color: 'var(--color-agent-niyati)' },
              { name: 'Iccha', role: 'Goal Extractor', color: 'var(--color-agent-iccha)' },
              { name: 'Karya', role: 'Planner & Scheduler', color: 'var(--color-agent-karya)' },
              { name: 'Karma', role: 'Executor & Action Taker', color: 'var(--color-agent-karma)' },
            ].map((agent) => (
              <div key={agent.name} className="flex items-center gap-2.5">
                <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: agent.color }} />
                <span className="text-sm" style={{ color: agent.color, fontFamily: 'var(--font-display)' }}>
                  {agent.name}
                </span>
                <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>— {agent.role}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </AnimatedPage>
  );
}