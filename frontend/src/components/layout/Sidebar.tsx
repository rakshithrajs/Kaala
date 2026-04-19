import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  MessageCircle,
  Target,
  CalendarClock,
  History,
  BarChart3,
  Settings,
  ChevronLeft,
  ChevronRight,
  Compass,
} from 'lucide-react';
import { useUIStore } from '@/stores/ui-store';
import { useWsStore } from '@/stores/ws-store';
import { cn } from '@/lib/utils';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/chat', icon: MessageCircle, label: 'Chat' },
  { to: '/goals', icon: Target, label: 'Goals' },
  { to: '/schedules', icon: CalendarClock, label: 'Schedules' },
  { to: '/history', icon: History, label: 'History' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar() {
  const collapsed = useUIStore((s) => s.sidebarCollapsed);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const connected = useWsStore((s) => s.connected);

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 h-full flex flex-col z-50 transition-all duration-300 border-r',
        collapsed ? 'w-16' : 'w-60'
      )}
      style={{
        background: 'linear-gradient(180deg, #0d0f18 0%, #0b0d14 100%)',
        borderColor: 'var(--color-border)',
      }}
    >
      {/* Brand — observatory instrument motif */}
      <div className="flex items-center gap-3 px-4 py-5 border-b" style={{ borderColor: 'var(--color-border)' }}>
        <div
          className="flex h-9 w-9 items-center justify-center rounded-md"
          style={{
            background: 'linear-gradient(135deg, rgba(200,164,78,0.2), rgba(200,164,78,0.05))',
            border: '1px solid rgba(200,164,78,0.3)',
          }}
        >
          <Compass className="h-4.5 w-4.5" style={{ color: 'var(--color-primary)' }} />
        </div>
        {!collapsed && (
          <div className="flex items-center gap-2.5">
            <span
              className="text-lg tracking-wide"
              style={{
                fontFamily: 'var(--font-display)',
                color: 'var(--color-primary)',
              }}
            >
              Kaala
            </span>
            <span
              className={cn('status-dot', connected ? 'connected' : 'disconnected')}
              title={connected ? 'Connected' : 'Disconnected'}
            />
          </div>
        )}
      </div>

      {/* Navigation — instrument panel labels */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-all duration-200',
                isActive
                  ? 'text-primary'
                  : 'text-text-muted hover:text-text'
              )
            }
            style={({ isActive }) => ({
              background: isActive
                ? 'linear-gradient(90deg, rgba(200,164,78,0.12), transparent)'
                : 'transparent',
              borderLeft: isActive
                ? '2px solid var(--color-primary)'
                : '2px solid transparent',
            })}
          >
            <Icon className="h-[18px] w-[18px] shrink-0" />
            {!collapsed && (
              <span style={{ fontFamily: 'var(--font-label)', fontWeight: 500 }}>{label}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle — brass accent */}
      <button
        onClick={toggleSidebar}
        className="flex items-center justify-center gap-2 border-t px-4 py-3 transition-colors"
        style={{ borderColor: 'var(--color-border)', color: 'var(--color-text-muted)' }}
        onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text)'}
        onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-muted)'}
      >
        {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        {!collapsed && <span className="text-[10px] uppercase tracking-widest" style={{ fontFamily: 'var(--font-label)' }}>Collapse</span>}
      </button>
    </aside>
  );
}