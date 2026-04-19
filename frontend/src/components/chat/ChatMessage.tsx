import { cn } from '@/lib/utils';
import { AGENT_COLORS } from '@/api/types';
import { ChatResultRenderer } from './ChatResultRenderer';
import type { ChatMessage as ChatMessageType } from '@/api/types';
import { User, Bot } from 'lucide-react';

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const agentColor = isUser ? 'var(--color-primary)' : (message.agent ? AGENT_COLORS[message.agent] : 'var(--color-text)');

  return (
    <div className={cn('flex gap-3', isUser ? 'flex-row-reverse' : 'flex-row')}>
      <div
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md"
        style={{
          backgroundColor: `color-mix(in srgb, ${agentColor} 15%, transparent)`,
          border: `1px solid color-mix(in srgb, ${agentColor} 25%, transparent)`,
        }}
      >
        {isUser ? (
          <User className="h-3.5 w-3.5" style={{ color: agentColor }} />
        ) : (
          <Bot className="h-3.5 w-3.5" style={{ color: agentColor }} />
        )}
      </div>
      <div
        className={cn('max-w-[75%] rounded-md px-4 py-3', isUser ? 'rounded-tr-sm' : 'rounded-tl-sm')}
        style={isUser
          ? {
              background: 'linear-gradient(135deg, rgba(200,164,78,0.12), rgba(200,164,78,0.06))',
              border: '1px solid rgba(200,164,78,0.2)',
            }
          : {
              background: 'var(--color-surface)',
              border: '1px solid var(--color-border)',
            }
        }
      >
        {message.agent && !isUser && (
          <p
            className="text-[10px] font-semibold uppercase tracking-wider mb-1.5"
            style={{ color: agentColor, fontFamily: 'var(--font-label)' }}
          >
            {message.agent}
          </p>
        )}
        <p
          className="text-sm whitespace-pre-wrap readout"
          style={{ lineHeight: '1.6' }}
        >
          {message.content}
        </p>
        {message.result && !isUser && <ChatResultRenderer result={message.result} />}
      </div>
    </div>
  );
}