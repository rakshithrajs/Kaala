import { useLocation } from 'react-router-dom';
import { Bell } from 'lucide-react';
import { useWsStore } from '@/stores/ws-store';

const pageNames: Record<string, string> = {
  '/': 'Dashboard',
  '/chat': 'Chat',
  '/goals': 'Goals',
  '/schedules': 'Schedules',
  '/history': 'History',
  '/analytics': 'Analytics',
  '/settings': 'Settings',
};

export function Header() {
  const location = useLocation();
  const lastEvent = useWsStore((s) => s.lastEvent);

  return (
    <header
      className="flex items-center justify-between border-b px-6 py-4"
      style={{
        background: 'rgba(11, 13, 20, 0.7)',
        backdropFilter: 'blur(12px)',
        borderColor: 'var(--color-border)',
      }}
    >
      <div className="flex items-center gap-3">
        <span
          className="label-text"
          style={{ color: 'var(--color-primary)' }}
        >
          {pageNames[location.pathname] ?? 'Kaala'}
        </span>
        <div
          className="brass-line flex-1"
          style={{ width: '60px' }}
        />
      </div>
      <div className="relative">
        <Bell
          className="h-4.5 w-4.5 transition-colors cursor-pointer"
          style={{ color: 'var(--color-text-muted)' }}
          onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text)'}
          onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-muted)'}
        />
        {lastEvent && (
          <span
            className="absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full animate-pulse"
            style={{ background: 'var(--color-accent-red)' }}
          />
        )}
      </div>
    </header>
  );
}