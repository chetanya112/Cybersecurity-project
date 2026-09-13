interface TrajectoryProps {
  pAttackCurve: number[];
  escalationScore: number;
  trajectoryStatus: string;
}

export function Trajectory({ pAttackCurve, escalationScore, trajectoryStatus }: TrajectoryProps) {
  const isEscalating = escalationScore >= 0.20;
  const isDeescalating = escalationScore <= -0.20;
  
  const statusColor = isEscalating ? 'text-danger' : isDeescalating ? 'text-success' : 'text-muted';

  return (
    <div className="flex gap-4">
      <div className="bg-surface border border-border p-4 rounded-md shadow-sm flex-1">
        <div className="text-[11px] text-muted font-medium mb-1">ESCALATION SCORE</div>
        <div className="text-4xl font-bold text-text mb-2">
          {escalationScore > 0 ? '+' : ''}{escalationScore.toFixed(2)}
        </div>
        <div className="text-[11px] text-muted font-medium mt-2">TRAJECTORY</div>
        <div className={`text-xl font-bold mt-1 ${statusColor}`}>
          {trajectoryStatus}
        </div>
        <div className="text-[11px] text-muted mt-3 leading-relaxed">
          Risk trend across W+1 → W+10.<br />
          Measures whether predicted attack probability is <em>rising</em> or <em>falling</em> over the forecast horizon.
        </div>
      </div>
      
      <div className="bg-surface border border-border p-4 rounded-md shadow-sm w-2/3">
        <div className="space-y-1.5">
          {pAttackCurve.map((p, i) => {
            const pct = p * 100;
            const bg = pct >= 70 ? 'bg-danger' : pct >= 30 ? 'bg-warning' : 'bg-success';
            const width = Math.max(Math.round(pct), 1);
            
            return (
              <div key={i} className="flex items-center gap-2">
                <span className="font-mono text-[11px] text-muted w-6">K{i+1}</span>
                <div className="flex-1 bg-[#151c28] rounded-[3px] h-1.5 overflow-hidden">
                  <div className={`h-full ${bg} rounded-[3px]`} style={{ width: `${width}%` }} />
                </div>
                <span className="font-mono text-xs text-text w-11 text-right">{pct.toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
