import type { LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description?: string;
}

export function EmptyState({ icon: Icon, title, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div
        className="rounded-md p-4 mb-4"
        style={{
          background: 'var(--color-surface)',
          border: '1px solid var(--color-border)',
        }}
      >
        <Icon className="h-8 w-8" style={{ color: 'var(--color-text-dim)' }} />
      </div>
      <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.125rem', color: 'var(--color-text)' }}>{title}</h3>
      {description && <p className="mt-1 text-sm" style={{ color: 'var(--color-text-muted)' }}>{description}</p>}
    </div>
  );
}