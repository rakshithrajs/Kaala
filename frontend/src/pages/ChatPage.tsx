import { ChatPanel } from '@/components/chat/ChatPanel';
import { AnimatedPage } from '@/components/shared/AnimatedPage';

export function ChatPage() {
  return (
    <AnimatedPage>
      <div className="flex items-center gap-3 mb-4">
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', color: 'var(--color-text)' }}>
          Reading
        </h2>
        <div className="brass-line flex-1" />
      </div>
      <ChatPanel />
    </AnimatedPage>
  );
}