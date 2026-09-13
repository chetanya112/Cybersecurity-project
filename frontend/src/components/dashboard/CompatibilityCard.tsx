import type { CompatibilityReport } from '../../api/types';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { CheckCircle2, AlertCircle, RefreshCw, Server, XCircle } from 'lucide-react';

interface Props {
  report: CompatibilityReport;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  onReset?: () => void;
}

export function CompatibilityCard({ report, onAnalyze, isAnalyzing, onReset }: Props) {
  const isNative = report.compatibility === 'NATIVE';
  const isUnsupported = report.compatibility === 'UNSUPPORTED';

  if (isUnsupported) {
    return (
      <Card highlight="red" className="max-w-2xl mx-auto my-8">
        <div className="flex items-center gap-3 mb-4">
          <XCircle className="text-danger" size={24} />
          <div className="text-lg font-bold tracking-[2px] text-danger uppercase">
            DATASET NOT SUPPORTED
          </div>
        </div>
        
        <div className="text-muted text-sm mb-6 leading-relaxed">
          {report.reason === 'TEMPORAL DATA REQUIRED' ? (
            <>
              <strong>TEMPORAL DATA REQUIRED</strong>
              <br/><br/>
              CHRONEX requires a valid time field to construct 30-second network states and perform temporal forecasting.
            </>
          ) : (
            <>
              CHRONEX could not construct its canonical network-state representation from this file.
              <br/><br/>
              <strong>Reason:</strong> {report.reason}
            </>
          )}
        </div>

        {report.missing_requirements && report.missing_requirements.length > 0 && (
          <div className="bg-[#151c28] rounded-md p-4 mb-6">
            <div className="text-xs font-bold text-text mb-3">MISSING SEMANTIC REQUIREMENTS:</div>
            <ul className="space-y-2 text-sm text-muted">
              {report.missing_requirements.map((req, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span className="text-danger">•</span> {req}
                </li>
              ))}
            </ul>
          </div>
        )}

        <button 
          className="px-4 py-2 bg-surface border border-border text-text hover:bg-border transition-colors rounded-md text-sm font-semibold w-full"
          onClick={() => onReset ? onReset() : window.location.reload()}
        >
          VIEW SUPPORTED FORMAT
        </button>
      </Card>
    );
  }

  return (
    <Card highlight={isNative ? 'green' : 'amber'} className="max-w-2xl mx-auto my-8">
      <div className="flex items-center gap-3 mb-6">
        {isNative ? (
          <CheckCircle2 className="text-success" size={24} />
        ) : (
          <RefreshCw className="text-warning" size={24} />
        )}
        <div className={`text-lg font-bold tracking-[2px] uppercase ${isNative ? 'text-success' : 'text-warning'}`}>
          {isNative ? 'NETWORK FLOW DETECTED' : 'SCHEMA ADAPTABLE'}
        </div>
      </div>

      <div className="space-y-4 mb-8">
        <div className="flex justify-between items-center py-2 border-b border-border">
          <span className="text-sm text-muted">Dataset compatibility</span>
          <Badge variant={isNative ? 'green' : 'amber'}>{isNative ? '100% NATIVE' : 'ADAPTED'}</Badge>
        </div>
        
        <div className="flex justify-between items-center py-2 border-b border-border">
          <span className="text-sm text-muted">Network-flow schema</span>
          <span className="text-success"><CheckCircle2 size={16} /></span>
        </div>
        <div className="flex justify-between items-center py-2 border-b border-border">
          <span className="text-sm text-muted">Timestamp</span>
          <span className={report.schema_flags?.timestamp ? 'text-success' : 'text-warning'}>
            {report.schema_flags?.timestamp ? (
              <CheckCircle2 size={16} />
            ) : (
              <div className="flex items-center gap-1.5 px-2 py-0.5 bg-warning/10 border border-warning/20 rounded">
                <span className="text-[10px] font-bold tracking-wider">SYNTHESIZED</span>
                <AlertCircle size={14} />
              </div>
            )}
          </span>
        </div>
        <div className="flex justify-between items-center py-2 border-b border-border">
          <span className="text-sm text-muted">Packet statistics</span>
          <span className={report.schema_flags?.packet_statistics ? 'text-success' : 'text-danger'}>
            {report.schema_flags?.packet_statistics ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
          </span>
        </div>
        <div className="flex justify-between items-center py-2 border-b border-border">
          <span className="text-sm text-muted">Byte statistics</span>
          <span className={report.schema_flags?.byte_statistics ? 'text-success' : 'text-danger'}>
            {report.schema_flags?.byte_statistics ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
          </span>
        </div>
        <div className="flex justify-between items-center py-2 border-b border-border">
          <span className="text-sm text-muted">Duration</span>
          <span className={report.schema_flags?.duration ? 'text-success' : 'text-danger'}>
            {report.schema_flags?.duration ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
          </span>
        </div>
        
        <div className="pt-2">
          <div className="flex justify-between items-center py-1">
            <span className="text-sm text-text font-medium">71 canonical features available</span>
            <span className="text-success"><CheckCircle2 size={16} /></span>
          </div>
          <div className="flex justify-between items-center py-1">
            <span className="text-sm text-text font-medium">30-second windows available</span>
            <span className="text-success"><CheckCircle2 size={16} /></span>
          </div>
          <div className="flex justify-between items-center py-1">
            <span className="text-sm text-text font-medium">10-window context ready ({report.usable_windows} detected)</span>
            <span className="text-success"><CheckCircle2 size={16} /></span>
          </div>
        </div>
      </div>

      <button 
        onClick={onAnalyze}
        disabled={isAnalyzing}
        className="w-full py-3 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-md text-sm font-bold tracking-widest flex items-center justify-center gap-2 transition-colors"
      >
        <Server size={18} />
        {isAnalyzing ? 'RUNNING INFERENCE...' : 'ANALYZE TRAFFIC'}
      </button>
    </Card>
  );
}
