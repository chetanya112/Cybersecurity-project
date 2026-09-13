import type { TemporalFeature, MitreMapping } from '../../api/types';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';

interface WhatChangedProps {
  evidence: TemporalFeature[];
  mitre: MitreMapping | null;
  isAttack: boolean;
}

export function WhatChanged({ evidence, mitre, isAttack }: WhatChangedProps) {
  return (
    <div className="flex gap-4">
      <div className="flex-1" id="explainability">
        <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2.5 uppercase">
          WHAT CHANGED? — TEMPORAL EVIDENCE
        </div>
        {evidence.length === 0 ? (
          <Card className="text-center py-6 text-muted text-sm">No evidence available.</Card>
        ) : (
          <Card className="space-y-3">
            {evidence.map((feat) => (
              <div key={feat.rank} className="mb-2">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-text">{feat.rank}. {feat.name}</span>
                  <span className="font-semibold" style={{ color: feat.color_code }}>
                    {feat.relative_change}
                  </span>
                </div>
                <div className="bg-[#151c28] h-[5px] rounded-[3px] overflow-hidden">
                  <div 
                    className="h-full rounded-[3px]" 
                    style={{ backgroundColor: feat.color_code, width: `${feat.evidence_percent}%` }} 
                  />
                </div>
                {isAttack && (
                  <div className="flex justify-between text-[10px] text-muted mt-1">
                    <span>SHAP {feat.shap_value >= 0 ? '+' : ''}{feat.shap_value.toFixed(2)}</span>
                    <span>Score {feat.score.toFixed(2)}</span>
                  </div>
                )}
              </div>
            ))}
          </Card>
        )}
      </div>
      
      <div className="flex-1" id="mitre">
        <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2.5 uppercase">
          MITRE ATT&CK MAPPING
        </div>
        {mitre ? (
          mitre.is_unclassified ? (
            <Card highlight="amber" className="h-full">
              <div className="text-[11px] text-muted font-medium mb-2">MITRE ATT&CK TACTIC</div>
              <Badge variant="amber">UNCLASSIFIED</Badge>
              <div className="text-xs text-muted mt-3">No SHAP-mapped features in top attribution.</div>
            </Card>
          ) : (
            <div className="bg-surface border border-border p-4 rounded-md shadow-sm h-full" style={{ borderTop: `3px solid ${mitre.color}` }}>
              <div className="text-[11px] text-muted font-medium">
                MITRE ATT&amp;CK TACTIC &nbsp; [ {mitre.tactic_id} ]
              </div>
              <div className="text-[22px] font-bold my-1.5" style={{ color: mitre.color }}>
                {mitre.emoji} {mitre.tactic}
              </div>
              <div className="text-xs text-muted mb-3">
                Confidence: <strong style={{ color: mitre.color }}>{mitre.confidence}</strong>
              </div>
              <div className="text-[11px] text-muted leading-relaxed">
                Based on K=1 SHAP attribution — model-aligned heuristic mapping.
              </div>
            </div>
          )
        ) : (
          <Card className="h-full flex items-center justify-center text-muted text-sm">
            NO ACTIVE ATTACK
          </Card>
        )}
      </div>
    </div>
  );
}
