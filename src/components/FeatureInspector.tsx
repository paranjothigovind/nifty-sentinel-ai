import React from 'react';
import { FeatureStats, FeatureRow } from '../types';
import { Layers, ArrowUpRight, ArrowDownRight, Activity } from 'lucide-react';

interface FeatureInspectorProps {
  stats: FeatureStats | null;
  features: FeatureRow[];
}

export const FeatureInspector: React.FC<FeatureInspectorProps> = ({ stats, features }) => {
  if (!stats || !stats.usable_samples) {
    return (
      <div className="p-8 text-center bg-[#121721] border border-[#1E2638] rounded-lg">
        <p className="text-sm text-[#8B949E]">
          No features generated yet. Please synchronize market data first.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Target Balance Banner (Dynamically Computed - No Hardcoding) */}
      <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#1E2638]">
          <div>
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#58A6FF]" />
              <h3 className="text-sm font-semibold text-[#E6EDF3]">
                Empirical Target Class Distribution (Unconditional Base Rate)
              </h3>
            </div>
            <p className="text-xs text-[#8B949E] mt-1">
              Binary Target: $Y_t = \mathbb&#123;I&#125;[Close_&#123;t+1&#125; &gt; Close_t]$. Computed dynamically from {stats.labeled_samples} labeled trading sessions.
            </p>
          </div>
          <div className="text-right">
            <span className="text-xs text-[#8B949E]">Historical Up Ratio</span>
            <div className="text-xl font-bold font-mono tabular-nums text-[#3FB950]">
              {stats.up_percentage.toFixed(2)}%
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
            <div className="flex items-center justify-between text-xs text-[#8B949E]">
              <span className="flex items-center gap-1.5 text-[#3FB950]">
                <ArrowUpRight className="w-3.5 h-3.5" /> Bullish Sessions (Close &gt; Prev)
              </span>
              <span className="font-mono tabular-nums text-[#E6EDF3] font-bold">
                {stats.up_count}
              </span>
            </div>
            <div className="w-full bg-[#1E2638] h-1.5 rounded-full mt-2 overflow-hidden">
              <div
                className="bg-[#3FB950] h-full"
                style={{ width: `${stats.up_percentage}%` }}
              />
            </div>
            <div className="text-[11px] text-[#8B949E] mt-1 font-mono tabular-nums text-right">
              {stats.up_percentage.toFixed(2)}% of sample
            </div>
          </div>

          <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
            <div className="flex items-center justify-between text-xs text-[#8B949E]">
              <span className="flex items-center gap-1.5 text-[#F85149]">
                <ArrowDownRight className="w-3.5 h-3.5" /> Bearish / Flat Sessions
              </span>
              <span className="font-mono tabular-nums text-[#E6EDF3] font-bold">
                {stats.down_count}
              </span>
            </div>
            <div className="w-full bg-[#1E2638] h-1.5 rounded-full mt-2 overflow-hidden">
              <div
                className="bg-[#F85149] h-full"
                style={{ width: `${100 - stats.up_percentage}%` }}
              />
            </div>
            <div className="text-[11px] text-[#8B949E] mt-1 font-mono tabular-nums text-right">
              {(100 - stats.up_percentage).toFixed(2)}% of sample
            </div>
          </div>

          <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
            <div className="flex items-center justify-between text-xs text-[#8B949E]">
              <span>Majority Baseline Accuracy</span>
              <span className="font-mono tabular-nums text-[#58A6FF] font-bold">
                {stats.up_percentage.toFixed(2)}%
              </span>
            </div>
            <p className="text-[11px] text-[#8B949E] mt-2 leading-relaxed">
              Any model predicting always "Up" achieves this accuracy blindly without having learned any genuine statistical edge.
            </p>
          </div>
        </div>
      </div>

      {/* Feature Engineering Definitions */}
      <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
        <h3 className="text-sm font-semibold text-[#E6EDF3] mb-3">
          Causal Feature Engine & Invariant Integrity
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
            <div className="font-semibold text-[#E6EDF3]">Lagged Price Returns</div>
            <div className="text-[11px] font-mono text-[#58A6FF] mt-0.5">return_1d, return_2d, return_5d, return_20d</div>
            <p className="text-[#8B949E] mt-1 leading-normal">
              Causal percentage return over 1, 2, 5, and 20 trading sessions. Captures short-term mean-reversion and momentum.
            </p>
          </div>

          <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
            <div className="font-semibold text-[#E6EDF3]">Moving Average Distance</div>
            <div className="text-[11px] font-mono text-[#58A6FF] mt-0.5">sma_20_ratio, sma_50_ratio</div>
            <p className="text-[#8B949E] mt-1 leading-normal">
              Normalized ratio $(Close_t / SMA_N - 1.0)$. Measures trend stretch and distance from intermediate baselines.
            </p>
          </div>

          <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
            <div className="font-semibold text-[#E6EDF3]">Oscillators & Volatility</div>
            <div className="text-[11px] font-mono text-[#58A6FF] mt-0.5">rsi_14, macd_diff, vol_20</div>
            <p className="text-[#8B949E] mt-1 leading-normal">
              14-day Wilder RSI, MACD 12/26/9 histogram, and 20-day annualized rolling volatility ($\sigma \times \sqrt&#123;252&#125;$).
            </p>
          </div>
        </div>
      </div>

      {/* Feature Data Table */}
      <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-[#E6EDF3]">
            Engineered Session Matrix (Recent 20 Sessions)
          </h3>
          <span className="text-xs text-[#8B949E] font-mono tabular-nums">
            Showing latest {Math.min(20, features.length)} rows
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
            <thead>
              <tr className="border-b border-[#1E2638] text-[#8B949E] bg-[#090C10]">
                <th className="py-2.5 px-3">Date</th>
                <th className="py-2.5 px-3">Close (₹)</th>
                <th className="py-2.5 px-3">Ret 1D</th>
                <th className="py-2.5 px-3">Ret 5D</th>
                <th className="py-2.5 px-3">SMA20 Ratio</th>
                <th className="py-2.5 px-3">RSI 14</th>
                <th className="py-2.5 px-3">MACD Diff</th>
                <th className="py-2.5 px-3">Vol 20 (Ann)</th>
                <th className="py-2.5 px-3 text-right">Target (t+1)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E2638]/60">
              {features.slice(-20).map((row, i) => {
                const isTargetUp = row.target === 1;
                const isPending = row.target === null || row.target === undefined;
                return (
                  <tr key={i} className="hover:bg-[#161B22] transition-colors">
                    <td className="py-2 px-3 text-[#E6EDF3]">{row.date}</td>
                    <td className="py-2 px-3">{row.close.toFixed(2)}</td>
                    <td className={`py-2 px-3 ${row.return_1d >= 0 ? 'text-[#3FB950]' : 'text-[#F85149]'}`}>
                      {(row.return_1d * 100).toFixed(2)}%
                    </td>
                    <td className={`py-2 px-3 ${row.return_5d >= 0 ? 'text-[#3FB950]' : 'text-[#F85149]'}`}>
                      {(row.return_5d * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 px-3">{(row.sma_20_ratio * 100).toFixed(2)}%</td>
                    <td className="py-2 px-3">{row.rsi_14.toFixed(1)}</td>
                    <td className="py-2 px-3">{row.macd_diff.toFixed(2)}</td>
                    <td className="py-2 px-3">{(row.vol_20 * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-right">
                      {isPending ? (
                        <span className="text-[#D29922] font-semibold">PENDING</span>
                      ) : isTargetUp ? (
                        <span className="text-[#3FB950] font-semibold">1 (UP)</span>
                      ) : (
                        <span className="text-[#F85149] font-semibold">0 (DOWN)</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
