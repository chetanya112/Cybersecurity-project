import { useState, useEffect } from 'react';
import { getDatasetInfo, getSequencePrediction } from '../api/client';
import type { PredictionResponse } from '../api/types';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';
import { Card } from '../components/common/Card';
import { Badge } from '../components/common/Badge';

export function Replay() {
  const [dataInfo, setDataInfo] = useState<{ num_sequences: number } | null>(null);
  const [seqIndex, setSeqIndex] = useState(0);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    getDatasetInfo().then(info => setDataInfo(info)).catch(console.error);
  }, []);

  useEffect(() => {
    let timer: number;
    if (isPlaying && dataInfo) {
      timer = window.setInterval(() => {
        setSeqIndex(prev => {
          if (prev >= dataInfo.num_sequences - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 3000); // 3 seconds per step
    }
    return () => clearInterval(timer);
  }, [isPlaying, dataInfo]);

  useEffect(() => {
    if (dataInfo) {
      getSequencePrediction(seqIndex)
        .then(res => setPrediction(res))
        .catch(console.error);
    }
  }, [seqIndex, dataInfo]);

  const handlePrev = () => setSeqIndex(p => Math.max(0, p - 1));
  const handleNext = () => setSeqIndex(p => dataInfo ? Math.min(dataInfo.num_sequences - 1, p + 1) : p);
  const handleReset = () => setSeqIndex(0);

  if (!dataInfo || !prediction) {
    return (
      <div className="h-full flex flex-col items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
        <div className="text-muted tracking-widest text-sm uppercase">Initializing Demo Mode...</div>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-[1200px] mx-auto min-h-screen flex flex-col">
      <div className="text-center py-6">
        <img 
          src="/assets/chronex-logo-clean.png" 
          alt="CHRONEX" 
          className="mx-auto max-w-[220px] w-full mb-3" 
          onError={(e) => {
            e.currentTarget.style.display = 'none';
            e.currentTarget.parentElement!.innerHTML = '<div class="text-5xl font-bold text-text mb-2">CHRONEX</div><div class="text-[11px] text-primary tracking-[4px] font-semibold">FORESEE. DEFEND. STAY AHEAD.</div>';
          }}
        />
        <div className="text-[11px] text-primary tracking-[4px] font-semibold">FORESEE. DEFEND. STAY AHEAD.</div>
      </div>

      <div className="flex-1 flex flex-col justify-center space-y-8">
        
        {/* Timeline */}
        <div>
          <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 uppercase text-center">ATTACK REPLAY TIMELINE</div>
          <div className="bg-surface border border-border rounded-md px-4 py-3 flex items-center justify-center overflow-x-auto">
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
                <div key={`fk-${i}`} className="min-w-[30px] h-[30px] rounded mx-0.5 flex items-center justify-center text-[10px] font-bold text-white shrink-0 transition-colors duration-500" style={{ backgroundColor: bg }}>
                  +{i+1}{isOnset ? '★' : ''}
                </div>
              );
            })}
          </div>
        </div>

        {/* Hero Cards */}
        <div className="grid grid-cols-2 gap-6">
          <Card highlight={prediction.is_attack ? 'red' : 'green'} className="text-center py-8">
            <div className="text-[11px] font-bold tracking-[2px] text-muted mb-2 uppercase">CURRENT STATE (K=1 NEXT-WINDOW)</div>
            <div className={`text-[52px] font-bold leading-none ${prediction.is_attack ? 'text-danger' : 'text-success'}`}>
              {(prediction.current.p_attack * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-muted mt-2 mb-4">Attack Probability</div>
            <Badge variant={prediction.is_attack ? 'red' : 'green'}>{prediction.current.pred_label}</Badge>
          </Card>

          <Card highlight={prediction.lead_time.status === 'EARLY WARNING' ? 'green' : 'none'} className="text-center py-8">
            <div className="text-[11px] font-bold tracking-[2px] text-muted mb-2 uppercase">FUTURE EARLY WARNING</div>
            {prediction.lead_time.status === 'EARLY WARNING' ? (
              <>
                <div className="text-[52px] font-bold leading-none text-success">
                  +{prediction.lead_time.lead_seconds}s
                </div>
                <div className="text-sm text-muted mt-2 mb-4">Forecast lead time before observed onset</div>
                <div className="flex gap-2 justify-center">
                  <Badge variant="blue">Trigger K={prediction.lead_time.trigger_k}</Badge>
                  <Badge variant="green">EARLY WARNING</Badge>
                </div>
              </>
            ) : prediction.lead_time.status === 'NO THREAT' ? (
              <div className="flex flex-col items-center justify-center h-[120px]">
                <div className="text-2xl font-bold text-muted mb-2">NO ATTACK IN HORIZON</div>
                <div className="text-sm text-muted">No ground-truth attack in W+1…W+10</div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[120px]">
                <div className="text-2xl font-bold text-warning mb-2">NO EARLY WARNING</div>
                <div className="text-sm text-muted">Attack onset at +{prediction.lead_time.lead_seconds}s was not predicted.</div>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Playback Controls Footer */}
      <div className="mt-12 pt-6 border-t border-border">
        <div className="flex items-center justify-center gap-6">
          <button onClick={handlePrev} className="p-3 bg-surface hover:bg-border rounded-full text-muted hover:text-text transition-colors">
            <SkipBack size={20} />
          </button>
          <button onClick={() => setIsPlaying(!isPlaying)} className="p-4 bg-primary hover:bg-primary/90 rounded-full text-white shadow-lg transition-transform hover:scale-105 active:scale-95">
            {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" className="ml-1" />}
          </button>
          <button onClick={handleNext} className="p-3 bg-surface hover:bg-border rounded-full text-muted hover:text-text transition-colors">
            <SkipForward size={20} />
          </button>
          <button onClick={handleReset} className="p-3 bg-surface hover:bg-border rounded-full text-muted hover:text-text transition-colors ml-4 border border-border">
            <RotateCcw size={16} />
          </button>
        </div>
        <div className="mt-6">
          <input 
            type="range" 
            min="0" 
            max={dataInfo.num_sequences - 1} 
            value={seqIndex}
            onChange={(e) => setSeqIndex(parseInt(e.target.value))}
            className="w-full accent-primary"
          />
          <div className="text-center text-xs text-muted mt-2">
            Sequence {seqIndex} of {dataInfo.num_sequences}
          </div>
        </div>
      </div>
    </div>
  );
}
