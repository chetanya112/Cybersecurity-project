import type { TopPrediction } from '../../api/types';

interface TopPredictionsProps {
  predictions: TopPrediction[];
  benignLabel: string;
}

export function TopPredictions({ predictions, benignLabel }: TopPredictionsProps) {
  return (
    <div className="mt-1 space-y-2">
      {predictions.map((pred, idx) => {
        const isBenign = pred.class_name === benignLabel;
        const color = isBenign ? 'bg-success' : 'bg-danger';
        const textColor = isBenign ? 'text-success' : 'text-danger';
        const width = Math.max(Math.round(pred.probability * 100), 1);
        
        return (
          <div key={idx} className="flex items-center justify-between py-1 border-b border-border last:border-0">
            <span className="text-xs text-text flex-1 truncate pr-2">{pred.class_name}</span>
            <div className="w-20 bg-surface border border-border rounded-[3px] h-1.5 mx-2 flex-shrink-0">
              <div className={`h-full rounded-[3px] ${color}`} style={{ width: `${width}%` }} />
            </div>
            <span className={`text-xs font-semibold w-11 text-right flex-shrink-0 ${textColor}`}>
              {(pred.probability * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
