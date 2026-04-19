import { cn } from '@/lib/utils';
import { STATUS_COLORS } from '@/api/types';

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const color = STATUS_COLORS[status] ?? 'var(--color-text-muted)';

  return (
    <span
      className={cn(
        'inline-flex items-center rounded px-2 py-0.5 readout',
        className
      )}
      style={{
        backgroundColor: `color-mix(in srgb, ${color} 10%, transparent)`,
        color: color,
        border: `1px solid color-mix(in srgb, ${color} 25%, transparent)`,
        fontSize: '0.65rem',
        fontFamily: 'var(--font-label)',
        fontWeight: 500,
        letterSpacing: '0.05em',
        textTransform: 'uppercase',
      }}
    >
      {status.replace('_', ' ')}
    </span>
  );
}