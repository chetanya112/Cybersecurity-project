import type { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  highlight?: 'red' | 'green' | 'blue' | 'amber' | 'none';
}

export function Card({ children, className = '', highlight = 'none' }: CardProps) {
  const highlights = {
    red: 'border-t-[3px] border-t-danger',
    green: 'border-t-[3px] border-t-success',
    blue: 'border-t-[3px] border-t-primary',
    amber: 'border-t-[3px] border-t-warning',
    none: 'border-t-transparent'
  };

  return (
    <div className={`bg-surface border border-border rounded-md p-4 shadow-sm ${highlights[highlight]} ${className}`}>
      {children}
    </div>
  );
}
