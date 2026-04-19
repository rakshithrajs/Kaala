import type { ChatResult } from '@/api/types';
import { CalendarClock, Zap, AlertTriangle, HelpCircle } from 'lucide-react';

interface ChatResultRendererProps {
  result: ChatResult;
}

export function ChatResultRenderer({ result }: ChatResultRendererProps) {
  switch (result.type) {
    case 'conversation':
      return null;

    case 'clarification':
      return (
        <div className="mt-3 pt-3 space-y-2" style={{ borderTop: '1px solid var(--color-border)' }}>
          <div className="flex items-center gap-2" style={{ color: 'var(--color-accent-amber)' }}>
            <HelpCircle className="h-3.5 w-3.5" />
            <span className="label-text" style={{ color: 'var(--color-accent-amber)' }}>Clarification needed</span>
          </div>
          {result.goals?.length > 0 && (
            <ul className="space-y-1 pl-1">
              {result.goals.map((goal, i) => (
                <li key={i} className="text-xs flex items-center gap-2 readout" style={{ color: 'var(--color-text-muted)' }}>
                  <span className="h-1 w-1 rounded-full" style={{ background: 'var(--color-accent-amber)' }} />
                  {goal}
                </li>
              ))}
            </ul>
          )}
        </div>
      );

    case 'immediate':
      return (
        <div className="mt-3 pt-3 space-y-2" style={{ borderTop: '1px solid var(--color-border)' }}>
          <div className="flex items-center gap-2" style={{ color: 'var(--color-accent-green)' }}>
            <Zap className="h-3.5 w-3.5" />
            <span className="label-text" style={{ color: 'var(--color-accent-green)' }}>Immediate action</span>
          </div>
          {result.results?.length > 0 && (
            <ul className="space-y-1 pl-1">
              {result.results.map((r, i) => (
                <li key={i} className="text-xs readout" style={{ color: 'var(--color-text-muted)' }}>
                  <span className="font-semibold" style={{ color: 'var(--color-text)' }}>{r.goal}</span>: {String(r.result)}
                </li>
              ))}
            </ul>
          )}
        </div>
      );

    case 'goals_scheduled':
      return (
        <div className="mt-3 pt-3 space-y-2" style={{ borderTop: '1px solid var(--color-border)' }}>
          <div className="flex items-center gap-2" style={{ color: 'var(--color-accent-teal)' }}>
            <CalendarClock className="h-3.5 w-3.5" />
            <span className="label-text" style={{ color: 'var(--color-accent-teal)' }}>
              {result.scheduled_prompts} prompt(s) scheduled
            </span>
          </div>
          {result.goals?.length > 0 && (
            <ul className="space-y-1 pl-1">
              {result.goals.map((goal, i) => (
                <li key={i} className="text-xs flex items-center gap-2 readout" style={{ color: 'var(--color-text-muted)' }}>
                  <span className="h-1 w-1 rounded-full" style={{ background: 'var(--color-accent-teal)' }} />
                  {goal}
                </li>
              ))}
            </ul>
          )}
        </div>
      );

    case 'executed':
      return (
        <div className="mt-3 pt-3 space-y-1" style={{ borderTop: '1px solid var(--color-border)' }}>
          <div className="flex items-center gap-2" style={{ color: 'var(--color-accent-orange)' }}>
            <Zap className="h-3.5 w-3.5" />
            <span className="label-text" style={{ color: 'var(--color-accent-orange)' }}>
              Executed via {result.tool ?? 'tool'}
            </span>
          </div>
          <p className="text-xs readout pl-1" style={{ color: 'var(--color-text-muted)' }}>{result.result}</p>
        </div>
      );

    case 'reminder':
      return (
        <div className="mt-3 pt-3 flex items-center gap-2" style={{ borderTop: '1px solid var(--color-border)', color: 'var(--color-accent-blue)' }}>
          <CalendarClock className="h-3.5 w-3.5" />
          <span className="label-text" style={{ color: 'var(--color-accent-blue)' }}>Reminder set</span>
        </div>
      );

    default:
      if ('error' in result && result.error) {
        return (
          <div className="mt-3 pt-3 flex items-center gap-2" style={{ borderTop: '1px solid var(--color-border)', color: 'var(--color-accent-red)' }}>
            <AlertTriangle className="h-3.5 w-3.5" />
            <span className="text-xs readout">{result.error}</span>
          </div>
        );
      }
      return null;
  }
}