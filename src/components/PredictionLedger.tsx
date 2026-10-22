import React from 'react';
import { PredictionRecord } from '../types';
import { Sparkles, Clock, CheckCircle2, XCircle, ArrowUpRight, ArrowDownRight, Compass } from 'lucide-react';

interface PredictionLedgerProps {
  latestPrediction: PredictionRecord | null;
  history: PredictionRecord[];
  onExplainWithGemini: () => void;
  isExplaining: boolean;
  onRefreshPrediction: () => void;
}

export const PredictionLedger: React.FC<PredictionLedgerProps> = ({
  latestPrediction,
  history,
  onExplainWithGemini,
  isExplaining,
  onRefreshPrediction,
}) => {
  return (
    <div className="space-y-6">
      {/* Latest Out-of-Sample Forecast Card */}
      {latestPrediction && !latestPrediction.error ? (
        <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#1E2638]">
            <div>
              <div className="flex items-center gap-2">
                <Compass className="w-4 h-4 text-[#58A6FF]" />
                <h3 className="text-sm font-semibold text-[#E6EDF3]">
                  Next Trading Session Directional Forecast (Out-of-Sample)
                </h3>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-[#8B949E] font-mono tabular-nums">
                <span>Signal Date: {latestPrediction.session_date}</span>
                <span aria-hidden="true">·</span>
                <span>Forecast Target Session: {latestPrediction.target_date}</span>
                <span aria-hidden="true">·</span>
                <span>Signal Close: ₹{latestPrediction.latest_close?.toFixed(2)}</span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={onExplainWithGemini}
                disabled={isExplaining}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#E6EDF3] bg-[#1E2638] hover:bg-[#28334b] border border-[#30363D] rounded transition-colors disabled:opacity-50 cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5 text-[#58A6FF]" />
                <span>{isExplaining ? 'Analyzing Regime...' : 'Interpret Regime with Gemini'}</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-5">
            {/* Predicted Direction & Probability */}
            <div className="p-4 bg-[#090C10] border border-[#1E2638] rounded">
              <div className="text-xs text-[#8B949E]">Model Directional Bias</div>
              <div className="flex items-center gap-2 mt-1">
                {latestPrediction.predicted_class === 1 ? (
                  <ArrowUpRight className="w-6 h-6 text-[#3FB950]" />
                ) : (
                  <ArrowDownRight className="w-6 h-6 text-[#F85149]" />
                )}
                <span
                  className={`text-lg font-bold ${
                    latestPrediction.predicted_class === 1 ? 'text-[#3FB950]' : 'text-[#F85149]'
                  }`}
                >
                  {latestPrediction.predicted_direction}
                </span>
              </div>

              <div className="mt-4 space-y-2 font-mono tabular-nums text-xs">
                <div>
                  <div className="flex justify-between text-[#8B949E] mb-1">
                    <span>P(Bullish Close &gt; Prev)</span>
                    <span className="text-[#3FB950] font-bold">
                      {((latestPrediction.p_up || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full bg-[#1E2638] h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-[#3FB950] h-full"
                      style={{ width: `${(latestPrediction.p_up || 0) * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[#8B949E] mb-1">
                    <span>P(Bearish / Flat Close)</span>
                    <span className="text-[#F85149] font-bold">
                      {(((latestPrediction.p_down || (1 - (latestPrediction.p_up || 0)))) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full bg-[#1E2638] h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-[#F85149] h-full"
                      style={{ width: `${(latestPrediction.p_down || (1 - (latestPrediction.p_up || 0))) * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="text-[11px] text-[#8B949E] font-mono mt-3">
                Confidence Spread |P - 0.5|: {((latestPrediction.confidence_delta || 0) * 100).toFixed(2)}%
              </div>
            </div>

            {/* Feature Contributions for this Session */}
            <div className="md:col-span-2 p-4 bg-[#090C10] border border-[#1E2638] rounded">
              <div className="text-xs text-[#8B949E] mb-2">Key Causal Drivers for this Prediction</div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono tabular-nums">
                {latestPrediction.feature_attributions?.slice(0, 6).map((f, i) => (
                  <div key={i} className="p-2 bg-[#121721] border border-[#1E2638] rounded flex justify-between">
                    <div>
                      <span className="text-[#E6EDF3] font-medium">{f.feature}</span>
                      <div className="text-[10px] text-[#8B949E]">
                        Val: {latestPrediction.features?.[f.feature] ?? 'N/A'}
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={f.direction === 'BULLISH' ? 'text-[#3FB950]' : 'text-[#F85149]'}>
                        {f.coefficient > 0 ? '+' : ''}
                        {f.coefficient.toFixed(3)}
                      </span>
                      <div className="text-[10px] text-[#8B949E]">OR: {f.odds_ratio}</div>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-[#8B949E] mt-3 font-mono">
                Stored in SQLite `model_predictions` at {latestPrediction.prediction_ts.slice(0, 19).replace('T', ' ')} UTC. Outcome will be audited upon market close.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-6 bg-[#121721] border border-[#1E2638] rounded-lg text-center">
          <p className="text-xs text-[#8B949E] mb-3">No active forecast generated yet.</p>
          <button
            onClick={onRefreshPrediction}
            className="px-3 py-1.5 text-xs text-white bg-[#58A6FF] rounded hover:bg-[#4094f7]"
          >
            Compute Next-Session Prediction
          </button>
        </div>
      )}

      {/* Historical Prediction Ledger Table */}
      <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-[#E6EDF3]">
              Persisted Prediction Ledger & Audit Log
            </h3>
            <p className="text-xs text-[#8B949E] mt-0.5">
              Permanently recorded forecasts compared with subsequent realized closing prices
            </p>
          </div>
          <span className="text-xs font-mono text-[#8B949E]">
            {history.length} Ledger Records
          </span>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-8 text-xs text-[#8B949E]">
            No historical predictions stored in SQLite yet. As sessions conclude, new forecasts are audited against realized market prices.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
              <thead>
                <tr className="border-b border-[#1E2638] text-[#8B949E] bg-[#090C10]">
                  <th className="py-2.5 px-3">Signal Date</th>
                  <th className="py-2.5 px-3">Target Date</th>
                  <th className="py-2.5 px-3">Signal Close</th>
                  <th className="py-2.5 px-3">Predicted P(Up)</th>
                  <th className="py-2.5 px-3">Call</th>
                  <th className="py-2.5 px-3">Realized Close</th>
                  <th className="py-2.5 px-3">Realized Dir</th>
                  <th className="py-2.5 px-3 text-right">Verification Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E2638]/60">
                {history.map((record, idx) => (
                  <tr key={idx} className="hover:bg-[#161B22] transition-colors">
                    <td className="py-2 px-3 text-[#E6EDF3]">{record.session_date}</td>
                    <td className="py-2 px-3 text-[#8B949E]">{record.target_date}</td>
                    <td className="py-2 px-3">{record.actual_close ? `₹${record.actual_close.toFixed(2)}` : '—'}</td>
                    <td className="py-2 px-3 font-semibold text-[#58A6FF]">
                      {(record.p_up * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 px-3">
                      {record.predicted_class === 1 ? (
                        <span className="text-[#3FB950] font-bold">UP</span>
                      ) : (
                        <span className="text-[#F85149] font-bold">DOWN</span>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      {record.target_close ? `₹${record.target_close.toFixed(2)}` : '—'}
                    </td>
                    <td className="py-2 px-3">
                      {record.actual_direction === 1 ? (
                        <span className="text-[#3FB950]">UP</span>
                      ) : record.actual_direction === 0 ? (
                        <span className="text-[#F85149]">DOWN</span>
                      ) : (
                        <span className="text-[#8B949E]">PENDING</span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {record.outcome_status === 'CORRECT' ? (
                        <span className="text-[#3FB950] font-bold flex items-center justify-end gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" /> CORRECT
                        </span>
                      ) : record.outcome_status === 'INCORRECT' ? (
                        <span className="text-[#F85149] font-bold flex items-center justify-end gap-1">
                          <XCircle className="w-3.5 h-3.5" /> INCORRECT
                        </span>
                      ) : (
                        <span className="text-[#D29922] font-semibold flex items-center justify-end gap-1">
                          <Clock className="w-3.5 h-3.5" /> PENDING CLOSE
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
