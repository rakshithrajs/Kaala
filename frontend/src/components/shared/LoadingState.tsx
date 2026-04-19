export function LoadingState({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-4 p-4">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="obsidian p-4 animate-pulse">
          <div className="h-4 w-1/3 rounded mb-3" style={{ background: 'var(--color-surface-hover)' }} />
          <div className="h-3 w-2/3 rounded mb-2" style={{ background: 'var(--color-surface-hover)' }} />
          <div className="h-3 w-1/2 rounded" style={{ background: 'var(--color-surface-hover)' }} />
        </div>
      ))}
    </div>
  );
}