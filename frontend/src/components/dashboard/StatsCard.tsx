import type { LucideIcon } from 'lucide-react';

interface StatsCardProps {
  icon: LucideIcon;
  label: string;
  value: number;
  color: string;
}

export function StatsCard({ icon: Icon, label, value, color }: StatsCardProps) {
  return (
    <div
      className="obsidian obsidian-hover p-5 transition-all duration-300"
      style={{
        borderLeft: `2px solid ${color}`,
      }}
    >
      <div className="flex items-center gap-3 mb-3">
        <div
          className="rounded-md p-2"
          style={{
            background: `color-mix(in srgb, ${color} 10%, transparent)`,
          }}
        >
          <Icon className="h-4 w-4" style={{ color }} />
        </div>
        <span className="label-text">{label}</span>
      </div>
      <p
        className="text-3xl readout"
        style={{ color, fontFamily: 'var(--font-body)', fontWeight: 700 }}
      >
        {value}
      </p>
    </div>
  );
}