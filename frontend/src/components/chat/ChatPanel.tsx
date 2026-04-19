import { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useChatStore } from '@/stores/chat-store';
import { useChat } from '@/hooks/useChat';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { Compass } from 'lucide-react';

export function ChatPanel() {
  const messages = useChatStore((s) => s.messages);
  const isSending = useChatStore((s) => s.isSending);
  const { sendMessage } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-[calc(100vh-10rem)]">
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full" style={{ color: 'var(--color-text-dim)' }}>
            <Compass
              className="h-14 w-14 mb-4"
              style={{ color: 'var(--color-primary)', opacity: 0.3 }}
            />
            <p style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', color: 'var(--color-text-muted)' }}>
              Begin a reading
            </p>
            <p className="label-text mt-2" style={{ color: 'var(--color-text-dim)' }}>
              Type a message below to consult the agents
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.15 }}
          >
            <ChatMessage message={msg} />
          </motion.div>
        ))}
        {isSending && (
          <motion.div
            animate={{ opacity: [0.3, 0.8, 0.3] }}
            transition={{ repeat: Infinity, duration: 2 }}
            className="flex items-center gap-2 text-sm"
            style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-body)' }}
          >
            <div className="flex gap-1">
              <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: 'var(--color-primary)', animation: 'brass-pulse 1.5s infinite' }} />
              <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: 'var(--color-primary)', animation: 'brass-pulse 1.5s infinite 0.3s' }} />
              <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: 'var(--color-primary)', animation: 'brass-pulse 1.5s infinite 0.6s' }} />
            </div>
            Processing...
          </motion.div>
        )}
        <div ref={bottomRef} />
      </div>
      <ChatInput onSend={sendMessage} disabled={isSending} />
    </div>
  );
}