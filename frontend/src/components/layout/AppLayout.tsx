import { Outlet } from 'react-router-dom';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useUIStore } from '@/stores/ui-store';
import { cn } from '@/lib/utils';

export function AppLayout() {
  useWebSocket();
  const collapsed = useUIStore((s) => s.sidebarCollapsed);

  return (
    <div className="flex h-screen overflow-hidden starfield" style={{ backgroundColor: 'var(--color-void)' }}>
      <Sidebar />
      <div
        className={cn('flex flex-1 flex-col overflow-hidden transition-all duration-300', collapsed ? 'ml-16' : 'ml-60')}
      >
        <Header />
        <main className="flex-1 overflow-y-auto p-6" style={{ color: 'var(--color-text)' }}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}