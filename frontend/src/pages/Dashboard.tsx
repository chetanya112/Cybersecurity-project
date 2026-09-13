import { useState, useEffect, useRef } from 'react';
import { RefreshCw } from 'lucide-react';
import type { DataSource } from '../App';
import { getDatasetInfo, getSequencePrediction, uploadCSV, analyzeTraffic } from '../api/client';
import type { PredictionResponse, CompatibilityReport } from '../api/types';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';
import { TopPredictions } from '../components/dashboard/TopPredictions';
import { ForecastChart } from '../components/charts/ForecastChart';
import { Trajectory } from '../components/dashboard/Trajectory';
import { WhatChanged } from '../components/dashboard/WhatChanged';
import { CompatibilityCard } from '../components/dashboard/CompatibilityCard';

type UploadStep = 'IDLE' | 'UPLOADING' | 'ANALYZING' | 'BUILDING' | 'VALIDATING' | 'READY';

export function Dashboard({ dataSource }: { dataSource: DataSource }) {
  const [dataInfo, setDataInfo] = useState<{ num_sequences: number } | null>(null);
  const [seqIndex, setSeqIndex] = useState(0);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Upload specific states
  const [uploadStep, setUploadStep] = useState<UploadStep>('IDLE');
  const [report, setReport] = useState<CompatibilityReport | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  useEffect(() => {
    if (dataSource === 'dataset') {
      getDatasetInfo().then(info => setDataInfo(info)).catch(console.error);
    }
  }, [dataSource]);

  useEffect(() => {
    if (dataSource === 'dataset') {
      setUploadStep('IDLE');
      setReport(null);
      loadSequence(seqIndex);
    } else {
      setPrediction(null);
      setError(null);
      setUploadStep('IDLE');
      setReport(null);
    }
  }, [seqIndex, dataSource]);

  const loadSequence = async (idx: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await getSequencePrediction(idx);
      setPrediction(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setError(null);
    setPrediction(null);
    setReport(null);
    setUploadProgress(0);
    setUploadStep('UPLOADING');
    
    try {
      const res = await uploadCSV(file, (progressEvent: any) => {
        if (progressEvent.total) {
          setUploadProgress(progressEvent.loaded / progressEvent.total);
        }
      });
      
      if (res.status === 'error') {
        setReport(res);
        setUploadStep('IDLE');
        return;
      }

      setUploadStep('ANALYZING');
      await new Promise(r => setTimeout(r, 400));
      setUploadStep('BUILDING');
      await new Promise(r => setTimeout(r, 400));
      setUploadStep('VALIDATING');
      await new Promise(r => setTimeout(r, 400));
      
      setReport(res);
      setUploadStep('READY');
      
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message);
      setUploadStep('IDLE');
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleAnalyzeTraffic = async () => {
    if (!report || !report.upload_id) return;
    
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeTraffic(report.upload_id);
      setPrediction(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderHeader = () => (
    <div className="flex justify-between items-center mb-6">
      <div>
        <span className="text-2xl font-bold text-text">CHRONEX</span>
        <span className="text-[15px] font-light text-muted ml-2">| Network Threat Intelligence</span>
      </div>
      <div className="flex gap-2">
        {prediction ? (
          <>
            <Badge variant="blue">{prediction.engine_label.toUpperCase()}</Badge>
            <Badge variant="blue">{prediction.feature_count} FEATURES</Badge>
          </>
        ) : (
          <>
            <Badge variant="blue">71 FEATURES</Badge>
            <Badge variant="blue">14 CLASSES</Badge>
          </>
        )}
        <Badge variant="gray">OFFLINE</Badge>
        {dataSource === 'upload' && (prediction || report) && (
          <button 
            onClick={() => {
              setPrediction(null);
              setReport(null);
              setUploadStep('IDLE');
              setError(null);
              if (fileInputRef.current) fileInputRef.current.value = '';
            }}
            className="ml-4 px-4 py-1.5 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30 transition-colors rounded text-xs font-bold tracking-wider flex items-center gap-2"
          >
            <RefreshCw size={14} /> NEW UPLOAD
          </button>
        )}
      </div>
    </div>
  );

  const renderUploadSteps = () => {
    if (uploadStep === 'IDLE') return null;
    
    const steps = [
      { id: 'UPLOADING', label: uploadStep === 'UPLOADING' && uploadProgress > 0 ? `UPLOADING (${(uploadProgress * 100).toFixed(0)}%)` : 'UPLOADED' },
      { id: 'ANALYZING', label: 'ANALYZING DATASET' },
      { id: 'BUILDING', label: 'BUILDING NETWORK STATES' },
      { id: 'VALIDATING', label: 'VALIDATING TEMPORAL HISTORY' },
      { id: 'READY', label: 'READY FOR ANALYSIS' }
    ];
    
    const currentIndex = steps.findIndex(s => s.id === uploadStep);
    
    return (
      <div className="flex items-center justify-center gap-4 my-8 text-xs font-bold tracking-[2px] uppercase">
        {steps.map((step, idx) => {
          let color = "text-muted";
          if (idx < currentIndex) color = "text-success";
          if (idx === currentIndex) color = "text-primary animate-pulse";
          
          return (
            <div key={step.id} className={`flex items-center gap-2 ${color}`}>
              {idx < currentIndex ? '✓' : idx === currentIndex ? '●' : '○'} {step.label}
            </div>
          );
        })}
      </div>
    );
  };

  if (loading && !prediction) {
    return (
      <div className="p-8 h-full flex flex-col items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
        <div className="text-muted tracking-widest text-sm uppercase">Analyzing Network Traffic...</div>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-[1600px] mx-auto">
      {renderHeader()}

      {dataSource === 'dataset' && dataInfo && (
        <div className="mb-6">
          <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 uppercase">DATASET SEQUENCE</div>
          <input 
            type="range" 
            min="0" 
            max={dataInfo.num_sequences - 1} 
            value={seqIndex}
            onChange={(e) => setSeqIndex(parseInt(e.target.value))}
            className="w-full max-w-md accent-primary"
          />
          <div className="text-xs text-muted mt-1">Sequence {seqIndex} of {dataInfo.num_sequences}</div>
        </div>
      )}

      {dataSource === 'upload' && !prediction && (
        <div className="mb-6">
          <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 uppercase">UPLOAD NETWORK TRAFFIC</div>
          <div className="flex items-center gap-4">
            <label className="cursor-pointer bg-primary hover:bg-primary/90 text-white font-bold py-2 px-6 rounded-md shadow-md transition-colors inline-flex items-center gap-2">
              <span>CHOOSE CSV</span>
              <input 
                type="file" 
                accept=".csv" 
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
            <span className="text-sm text-muted">Drop a compatible network-flow CSV to begin.</span>
          </div>
          
          {renderUploadSteps()}
          
          {report && report.status === 'error' && report.error_code === 'UPLOAD_TOO_LARGE' && (
            <Card highlight="red" className="max-w-2xl mx-auto my-8">
              <div className="text-xl font-bold tracking-[2px] text-danger mb-4 uppercase">
                UPLOAD TOO LARGE
              </div>
              <div className="text-sm text-muted mb-6">
                {report.reason}
                <br/><br/>
                CHRONEX requires network-flow CSVs under the configured limit to ensure memory-safe processing.
                Please choose a smaller network-flow CSV or configure a higher limit if resources permit.
              </div>
              <button 
                onClick={() => { setReport(null); setUploadStep('IDLE'); }}
                className="w-full py-2 bg-surface border border-border hover:bg-border/50 text-text rounded-md text-sm font-semibold transition-colors"
              >
                CHOOSE ANOTHER FILE
              </button>
            </Card>
          )}

          {report && (!report.error_code || report.error_code !== 'UPLOAD_TOO_LARGE') && (
            <CompatibilityCard 
              report={report} 
              onAnalyze={handleAnalyzeTraffic}
              isAnalyzing={loading}
              onReset={() => {
                setPrediction(null);
                setReport(null);
                setUploadStep('IDLE');
                setError(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
            />
          )}
        </div>
      )}

      {error && (
        <div className="bg-danger/10 border border-danger/30 text-danger p-4 rounded-md mb-6">
          <div className="font-bold mb-1">INFERENCE FAILED</div>
          <div className="text-sm">{error}</div>
        </div>
      )}

      {prediction && (
        <div className="space-y-6">
          {/* Overview Hero */}
          <div id="overview">
            <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 uppercase">OVERVIEW</div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card highlight={prediction.is_attack ? 'red' : 'green'}>
                <div className="text-[11px] text-muted font-medium mb-1">NEXT-WINDOW RISK — K=1</div>
                <div className={`text-4xl font-bold ${prediction.is_attack ? 'text-danger' : 'text-success'} mb-2`}>
                  {(prediction.current.p_attack * 100).toFixed(1)}%
                </div>
                <div className="mb-2">
                  <Badge variant={prediction.is_attack ? 'red' : 'green'}>{prediction.current.pred_label}</Badge>
                </div>
                <div className="text-xs text-muted mt-1">
                  Reliability: <strong className={
                    prediction.current.reliability === 'HIGH' ? 'text-success' : 
                    prediction.current.reliability === 'MODERATE' ? 'text-warning' : 'text-danger'
                  }>{prediction.current.reliability}</strong>
                </div>
              </Card>

              <Card>
                <div className="text-[11px] text-muted font-medium mb-1">TOP PREDICTIONS</div>
                <div className="text-xs text-muted mb-2">K=1 class probabilities</div>
                <TopPredictions predictions={prediction.current.top_predictions} benignLabel="Benign" />
              </Card>

              <Card highlight="blue">
                <div className="text-[11px] text-muted font-medium mb-1">FORECAST WINDOW</div>
                <div className="text-2xl font-bold text-text mb-2">30 s → 5 min</div>
                <div className="text-xs text-muted mb-2">W+1 = 30 sec │ W+10 = 300 sec</div>
                <div className="text-[11px] text-muted mt-2">10 consecutive future states predicted simultaneously.</div>
              </Card>

              <Card highlight={
                prediction.lead_time.status === 'EARLY WARNING' ? 'green' : 
                prediction.lead_time.status === 'MISSED' ? 'amber' : 'none'
              }>
                <div className="text-[11px] text-muted font-medium mb-1">FUTURE EARLY WARNING</div>
                {prediction.lead_time.status === 'EARLY WARNING' ? (
                  <>
                    <div className="text-3xl font-bold text-success mt-1 mb-2">+{prediction.lead_time.lead_seconds}s</div>
                    <div className="mb-2">
                      <Badge variant="blue">Trigger K={prediction.lead_time.trigger_k}</Badge>
                    </div>
                    <div className="text-xs text-muted mt-1">
                      Lead time before observed attack onset.<br/>
                      K=1 next: <strong>{prediction.current.pred_label}</strong>
                    </div>
                  </>
                ) : prediction.lead_time.status === 'NO THREAT' ? (
                  <>
                    <div className="text-2xl font-bold text-muted mt-1 mb-2">No Threat</div>
                    <div className="text-xs text-muted">No attack in W+1…W+10</div>
                  </>
                ) : prediction.lead_time.status === 'MISSED' && prediction.forecasts.some(f => f.posthoc_ground_truth) ? (
                  <>
                    <div className="text-2xl font-bold text-warning mt-1 mb-2">Missed</div>
                    <div className="text-xs text-muted">Attack at +{prediction.lead_time.lead_seconds}s not predicted before onset.</div>
                  </>
                ) : (
                  <>
                    <div className="text-2xl font-bold text-muted mt-1 mb-2">N/A</div>
                    <div className="text-xs text-muted">Lead-time evaluation requires observed ground-truth labels (native dataset mode).</div>
                  </>
                )}
              </Card>
            </div>
          </div>

          <hr className="border-border my-6" />

          {/* Future Threat Forecast */}
          <div id="forecast" className="scroll-mt-6">
            <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 uppercase">FUTURE THREAT FORECAST</div>
            
            <div className="bg-surface border border-border rounded-md px-4 py-2 flex items-center mb-4 overflow-x-auto">
              <div className="text-[9px] font-bold tracking-[1.5px] text-muted mx-2 shrink-0">OBSERVED</div>
              {Array.from({ length: 10 }).map((_, i) => (
                <div key={`w-${i}`} className="min-w-[30px] h-[30px] rounded bg-[#1e2d44] mx-0.5 flex items-center justify-center text-[10px] font-bold text-white shrink-0" title={`W-${9-i}`}>
                  {i < 9 ? `W${-9+i}` : 'W0'}
                </div>
              ))}
              <div className="text-[#2d3748] text-[22px] mx-2 shrink-0">│</div>
              <div className="text-[9px] font-bold tracking-[1.5px] text-muted mx-2 shrink-0">FORECAST</div>
              {prediction.trajectory.p_attack_curve.map((p, i) => {
                const isOnset = prediction.lead_time.onset_index === i;
                const bg = p >= 0.70 ? '#ef4444' : p >= 0.30 ? '#f59e0b' : '#10b981';
                return (
                  <div key={`fk-${i}`} className="min-w-[30px] h-[30px] rounded mx-0.5 flex items-center justify-center text-[10px] font-bold text-white shrink-0" style={{ backgroundColor: bg }}>
                    +{i+1}{isOnset ? '★' : ''}
                  </div>
                );
              })}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {prediction.forecasts.map((f) => (
                <Card key={f.k} highlight={f.pred_class !== 0 ? 'red' : 'green'}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-base font-bold text-textMuted">K={f.k}</span>
                    <span className="text-[11px] font-medium text-muted">+{f.horizon_seconds}s</span>
                  </div>
                  <div className="text-[22px] font-semibold text-text mb-1">
                    {(f.p_attack * 100).toFixed(1)}%
                  </div>
                  <div className="mb-3">
                    <Badge variant={f.pred_class !== 0 ? 'red' : 'green'}>{f.pred_label}</Badge>
                  </div>
                  {f.posthoc_ground_truth && (
                    <div className="border-t border-border pt-2 mt-2">
                      <div className="text-[11px] text-muted font-medium mb-0.5">Ground Truth (Post-Hoc)</div>
                      <div className="text-xs text-text">
                        {f.posthoc_ground_truth} <span className={f.is_correct ? 'text-success' : 'text-danger'}>{f.is_correct ? '✅ Match' : '❌ Mismatch'}</span>
                      </div>
                    </div>
                  )}
                </Card>
              ))}
            </div>

            <ForecastChart 
              pAttackCurve={prediction.trajectory.p_attack_curve} 
              onsetIndex={prediction.lead_time.onset_index} 
            />
          </div>

          <hr className="border-border my-6" />

          {/* Trajectory */}
          <div id="trajectory" className="scroll-mt-6">
            <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 uppercase">ATTACK TRAJECTORY</div>
            <Trajectory 
              pAttackCurve={prediction.trajectory.p_attack_curve}
              escalationScore={prediction.trajectory.escalation_score}
              trajectoryStatus={prediction.trajectory.trajectory_status}
            />
          </div>

          <hr className="border-border my-6" />

          {/* Evidence and MITRE */}
          <WhatChanged 
            evidence={prediction.evidence}
            mitre={prediction.mitre}
            isAttack={prediction.is_attack}
          />

        </div>
      )}
    </div>
  );
}
