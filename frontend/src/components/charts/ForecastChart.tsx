import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceLine, Tooltip } from 'recharts';

interface ForecastChartProps {
  pAttackCurve: number[];
  onsetIndex: number | null;
}

export function ForecastChart({ pAttackCurve, onsetIndex }: ForecastChartProps) {
  const data = pAttackCurve.map((p, idx) => ({
    k: idx + 1,
    pAttack: p * 100
  }));

  return (
    <div className="h-[220px] w-full mt-3">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a2234" vertical={false} />
          <XAxis 
            dataKey="k" 
            tick={{ fill: '#475569', fontSize: 11 }} 
            tickLine={{ stroke: '#1a2234' }}
            axisLine={{ stroke: '#1a2234' }}
            tickFormatter={(val) => `K=${val}`}
          />
          <YAxis 
            domain={[0, 100]} 
            tick={{ fill: '#475569', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(val) => `${val}%`}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#0d1117', borderColor: '#1a2234', borderRadius: '6px' }}
            itemStyle={{ color: '#0ea5e9' }}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(value: any) => [`${value.toFixed(1)}%`, 'P(Attack)']}
            labelFormatter={(label) => `W+${label}`}
          />
          {onsetIndex !== null && onsetIndex !== -1 && (
            <ReferenceLine 
              x={onsetIndex + 1} 
              stroke="#ef4444" 
              strokeDasharray="4 4" 
              label={{ position: 'top', value: 'GT Onset', fill: '#ef4444', fontSize: 11 }} 
            />
          )}
          <Line 
            type="monotone" 
            dataKey="pAttack" 
            stroke="#0ea5e9" 
            strokeWidth={2.5}
            dot={{ r: 4, fill: '#080c14', stroke: '#0ea5e9', strokeWidth: 2 }}
            activeDot={{ r: 6, fill: '#0ea5e9' }}
            isAnimationActive={true}
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
