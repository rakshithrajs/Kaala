import { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div
      className="flex items-end gap-3 p-3 mt-4 rounded-md"
      style={{
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
      }}
    >
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Enter a reading..."
        disabled={disabled}
        rows={1}
        className="flex-1 resize-none rounded-md px-4 py-2.5 text-sm readout"
        style={{
          backgroundColor: 'var(--color-abyss)',
          color: 'var(--color-text)',
          border: '1px solid var(--color-border)',
          outline: 'none',
          fontFamily: 'var(--font-body)',
        }}
        onFocus={(e) => e.currentTarget.style.borderColor = 'var(--color-primary)'}
        onBlur={(e) => e.currentTarget.style.borderColor = 'var(--color-border)'}
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || !input.trim()}
        className="flex h-10 w-10 items-center justify-center rounded-md transition-all"
        style={{
          background: disabled || !input.trim() ? 'var(--color-surface-hover)' : 'var(--color-primary)',
          color: disabled || !input.trim() ? 'var(--color-text-dim)' : 'var(--color-void)',
        }}
      >
        <Send className="h-4 w-4" />
      </button>
    </div>
  );
}