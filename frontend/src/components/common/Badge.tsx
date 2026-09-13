import type { ReactNode } from 'react';

interface BadgeProps {
  children: ReactNode;
  variant?: 'red' | 'green' | 'blue' | 'amber' | 'gray';
}

export function Badge({ children, variant = 'gray' }: BadgeProps) {
  const variants = {
    red: 'bg-danger/10 text-danger border-danger/25',
    green: 'bg-success/10 text-success border-success/25',
    blue: 'bg-primary/10 text-primary border-primary/25',
    amber: 'bg-warning/10 text-warning border-warning/25',
    gray: 'bg-muted/10 text-textMuted border-muted/25'
  };

  return (
    <span className={`inline-block px-2.5 py-0.5 border rounded-[3px] text-[11px] font-semibold whitespace-nowrap ${variants[variant]}`}>
      {children}
    </span>
  );
}
