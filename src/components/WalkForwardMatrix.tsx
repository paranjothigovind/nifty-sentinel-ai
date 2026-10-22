import React, { useState } from 'react';
import { BacktestResult, WindowComparison } from '../types';
import { GitCompare, TrendingUp, CheckCircle, BarChart3, HelpCircle, AlertTriangle, ShieldCheck, Scale } from 'lucide-react';

interface WalkForwardMatrixProps {
  comparison: WindowComparison | null;
  onRunBacktest: (minTrain: number) => void;
  isRunning: boolean;
}

export const WalkForwardMatrix: React.FC<WalkForwardMatrixProps> = ({
  comparison,
  onRunBacktest,
  isRunning,
}) => {
  const [selectedView, setSelectedView] = useState<'COMMON' | 'FULL_COMPARE' | '252' | '750'>('COMMON');

  if (!comparison || (!comparison.window_252 && !comparison.window_750)) {
    return (
      <div className="p-8 text-center bg-[#121721] border border-[#1E2638] rounded-lg">
        <GitCompare className="w-8 h-8 text-[#58A6FF] mx-auto mb-2 opacity-60" />
        <h3 className="text-base font-semibold text-[#E6EDF3]">Walk-Forward Out-of-Sample Backtest</h3>
        <p className="text-xs text-[#8B949E] max-w-md mx-auto mt-1 mb-4">
          Execute chronological cross-validation with expanding training windows to test predictability with zero lookahead leakage.
        </p>
        <button
          onClick={() => onRunBacktest(252)}
          disabled={isRunning}
          className="px-4 py-2 text-xs font-medium text-white bg-[#58A6FF] hover:bg-[#4094f7] disabled:opacity-50 rounded transition-colors"
        >
          {isRunning ? 'Computing Walk-Forward Backtest...' : 'Execute Walk-Forward Evaluation'}
        </button>
      </div>
    );
  }

  const w252 = comparison.window_252;
  const w750 = comparison.window_750;
  const common = comparison.common_dates_comparison;
  const refitSessions = comparison.refit_interval || 60;

  // Active result for single view
  const activeResult: BacktestResult = selectedView === '750' ? w750 : w252;

  return (
    <div className="space-y-6">
      {/* Configuration & Fair Comparison Header */}
      <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#1E2638]">
          <div>
            <div className="flex items-center gap-2">
              <GitCompare className="w-4 h-4 text-[#58A6FF]" />
              <h3 className="text-sm font-semibold text-[#E6EDF3]">
                Chronological Walk-Forward Backtest & Fair Seed Comparison
              </h3>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-[#8B949E] mt-1 font-mono tabular-nums">
              <span>Expanding Window (Zero Lookahead Leakage)</span>
              <span aria-hidden="true">·</span>
              <span className="text-[#58A6FF]">Refit Cadence: {refitSessions} sessions (~quarterly)</span>
              <span aria-hidden="true">·</span>
              <span>L2 Logistic Regression vs Unconditional Majority Baseline</span>
            </div>
          </div>

          {/* Segmented View Switcher */}
          <div className="flex items-center gap-1 p-0.5 bg-[#090C10] border border-[#1E2638] rounded">
            <button
              onClick={() => setSelectedView('COMMON')}
              className={`px-2.5 py-1 text-xs font-medium transition-colors rounded ${
                selectedView === 'COMMON' ? 'bg-[#1E2638] text-[#E6EDF3]' : 'text-[#8B949E] hover:text-[#E6EDF3]'
              }`}
            >
              Fair Common-Dates ({common?.common_sample_count || 434}d)
            </button>
            <button
              onClick={() => setSelectedView('FULL_COMPARE')}
              className={`px-2.5 py-1 text-xs font-medium transition-colors rounded ${
                selectedView === 'FULL_COMPARE' ? 'bg-[#1E2638] text-[#E6EDF3]' : 'text-[#8B949E] hover:text-[#E6EDF3]'
              }`}
            >
              Full Lifespans Side-by-Side
            </button>
            <button
              onClick={() => setSelectedView('252')}
              className={`px-2.5 py-1 text-xs font-medium transition-colors rounded ${
                selectedView === '252' ? 'bg-[#1E2638] text-[#E6EDF3]' : 'text-[#8B949E] hover:text-[#E6EDF3]'
              }`}
            >
              252 Seed (1Y)
            </button>
            <button
              onClick={() => setSelectedView('750')}
              className={`px-2.5 py-1 text-xs font-medium transition-colors rounded ${
                selectedView === '750' ? 'bg-[#1E2638] text-[#E6EDF3]' : 'text-[#8B949E] hover:text-[#E6EDF3]'
              }`}
            >
              750 Seed (3Y)
            </button>
          </div>
        </div>

        {/* 1. FAIR COMMON-DATES VIEW (Audit Requirement 4) */}
        {selectedView === 'COMMON' && common && (
          <div className="mt-4 space-y-4">
            <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded flex items-center justify-between text-xs font-mono tabular-nums">
              <div className="flex items-center gap-2">
                <Scale className="w-4 h-4 text-[#58A6FF]" />
                <span className="text-[#E6EDF3] font-semibold">Strict Out-of-Sample Date Matching</span>
              </div>
              <div className="text-[#8B949E]">
                Sessions: <span className="text-[#E6EDF3] font-bold">{common.common_sample_count}</span> · Dates:{' '}
                <span className="text-[#E6EDF3]">{common.common_start_date}</span> to{' '}
                <span className="text-[#E6EDF3]">{common.common_end_date}</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                <thead>
                  <tr className="border-b border-[#1E2638] text-[#8B949E] bg-[#090C10]">
                    <th className="py-2.5 px-3">Model Configuration</th>
                    <th className="py-2.5 px-3">Directional Accuracy (95% Wilson CI)</th>
                    <th className="py-2.5 px-3">Balanced Acc</th>
                    <th className="py-2.5 px-3">Brier Score Loss</th>
                    <th className="py-2.5 px-3">Log Loss</th>
                    <th className="py-2.5 px-3 text-right">Beats Common Baseline</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E2638]/60">
                  <tr className="hover:bg-[#161B22]">
                    <td className="py-2.5 px-3 text-[#E6EDF3] font-semibold">
                      252-Session Seed (Trained &gt; 1Y, Matched Common Slice)
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="font-bold text-[#E6EDF3]">
                        {(common.models.seed_252.accuracy * 100).toFixed(2)}%
                      </span>{' '}
                      <span className="text-[11px] text-[#8B949E]">
                        [{(common.models.seed_252.accuracy_ci_95[0] * 100).toFixed(1)}% -{' '}
                        {(common.models.seed_252.accuracy_ci_95[1] * 100).toFixed(1)}%]
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      {(common.models.seed_252.balanced_accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-[#3FB950]">
                      {common.models.seed_252.brier_score.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3">{common.models.seed_252.log_loss.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right">
                      {common.models.seed_252.beats_baseline ? (
                        <span className="text-[#3FB950] font-bold">YES</span>
                      ) : (
                        <span className="text-[#8B949E]">NO</span>
                      )}
                    </td>
                  </tr>

                  <tr className="hover:bg-[#161B22]">
                    <td className="py-2.5 px-3 text-[#E6EDF3] font-semibold">
                      750-Session Seed (Trained &gt; 3Y, Full Out-of-Sample)
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="font-bold text-[#E6EDF3]">
                        {(common.models.seed_750.accuracy * 100).toFixed(2)}%
                      </span>{' '}
                      <span className="text-[11px] text-[#8B949E]">
                        [{(common.models.seed_750.accuracy_ci_95[0] * 100).toFixed(1)}% -{' '}
                        {(common.models.seed_750.accuracy_ci_95[1] * 100).toFixed(1)}%]
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      {(common.models.seed_750.balanced_accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-[#3FB950]">
                      {common.models.seed_750.brier_score.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3">{common.models.seed_750.log_loss.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right">
                      {common.models.seed_750.beats_baseline ? (
                        <span className="text-[#3FB950] font-bold">YES</span>
                      ) : (
                        <span className="text-[#8B949E]">NO</span>
                      )}
                    </td>
                  </tr>

                  <tr className="hover:bg-[#161B22] text-[#8B949E]">
                    <td className="py-2.5 px-3 font-semibold">
                      Unconditional Baseline (Evaluated on Common Dates)
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-[#E6EDF3]">
                      {(common.models.baseline.accuracy * 100).toFixed(2)}%{' '}
                      <span className="text-[11px] text-[#8B949E]">
                        [{(common.models.baseline.accuracy_ci_95[0] * 100).toFixed(1)}% -{' '}
                        {(common.models.baseline.accuracy_ci_95[1] * 100).toFixed(1)}%]
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      {(common.models.baseline.balanced_accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3">{common.models.baseline.brier_score.toFixed(4)}</td>
                    <td className="py-2.5 px-3">{common.models.baseline.log_loss.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right">—</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded text-xs text-[#8B949E] space-y-1">
              <div>
                <span className="text-[#E6EDF3] font-semibold">Seed Comparison Result:</span>{' '}
                {common.head_to_head.accuracy_diff_750_vs_252 > 0
                  ? `750-session seed leads 252-session seed by +${(common.head_to_head.accuracy_diff_750_vs_252 * 100).toFixed(2)}% accuracy on common dates.`
                  : common.head_to_head.accuracy_diff_750_vs_252 < 0
                  ? `252-session seed leads 750-session seed by +${Math.abs(common.head_to_head.accuracy_diff_750_vs_252 * 100).toFixed(2)}% accuracy on common dates.`
                  : 'Both seeds achieved identical accuracy on the common out-of-sample period.'}
              </div>
              <div className="text-[11px]">
                {common.head_to_head.fair_comparison_note} Both models are tested on observations strictly unknown when their training took place.
              </div>
            </div>
          </div>
        )}

        {/* 2. FULL LIFESPAN SIDE-BY-SIDE VIEW */}
        {selectedView === 'FULL_COMPARE' && (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
              <thead>
                <tr className="border-b border-[#1E2638] text-[#8B949E] bg-[#090C10]">
                  <th className="py-2.5 px-3">Quantitative Metric</th>
                  <th className="py-2.5 px-3">252-Session Seed ({w252.evaluated_test_sessions} sessions)</th>
                  <th className="py-2.5 px-3">750-Session Seed ({w750.evaluated_test_sessions} sessions)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E2638]/60">
                <tr className="hover:bg-[#161B22]">
                  <td className="py-2 px-3 text-[#E6EDF3] font-semibold">Evaluation Horizon</td>
                  <td className="py-2 px-3 text-[#8B949E]">{w252.test_start_date} to {w252.test_end_date}</td>
                  <td className="py-2 px-3 text-[#8B949E]">{w750.test_start_date} to {w750.test_end_date}</td>
                </tr>
                <tr className="hover:bg-[#161B22]">
                  <td className="py-2 px-3 text-[#E6EDF3] font-semibold">Directional Accuracy (95% CI)</td>
                  <td className="py-2 px-3">
                    <span className="font-bold text-[#E6EDF3]">{(w252.metrics.model.accuracy * 100).toFixed(2)}%</span>{' '}
                    <span className="text-[11px] text-[#8B949E]">
                      [{(w252.metrics.model.accuracy_ci_95?.[0] ? w252.metrics.model.accuracy_ci_95[0] * 100 : 0).toFixed(1)}% -{' '}
                      {(w252.metrics.model.accuracy_ci_95?.[1] ? w252.metrics.model.accuracy_ci_95[1] * 100 : 0).toFixed(1)}%]
                    </span>
                  </td>
                  <td className="py-2 px-3">
                    <span className="font-bold text-[#E6EDF3]">{(w750.metrics.model.accuracy * 100).toFixed(2)}%</span>{' '}
                    <span className="text-[11px] text-[#8B949E]">
                      [{(w750.metrics.model.accuracy_ci_95?.[0] ? w750.metrics.model.accuracy_ci_95[0] * 100 : 0).toFixed(1)}% -{' '}
                      {(w750.metrics.model.accuracy_ci_95?.[1] ? w750.metrics.model.accuracy_ci_95[1] * 100 : 0).toFixed(1)}%]
                    </span>
                  </td>
                </tr>
                <tr className="hover:bg-[#161B22]">
                  <td className="py-2 px-3 text-[#E6EDF3] font-semibold">Majority Baseline Accuracy (95% CI)</td>
                  <td className="py-2 px-3 text-[#8B949E]">
                    {(w252.metrics.baseline.accuracy * 100).toFixed(2)}%{' '}
                    <span className="text-[11px]">
                      [{(w252.metrics.baseline.accuracy_ci_95?.[0] ? w252.metrics.baseline.accuracy_ci_95[0] * 100 : 0).toFixed(1)}% -{' '}
                      {(w252.metrics.baseline.accuracy_ci_95?.[1] ? w252.metrics.baseline.accuracy_ci_95[1] * 100 : 0).toFixed(1)}%]
                    </span>
                  </td>
                  <td className="py-2 px-3 text-[#8B949E]">
                    {(w750.metrics.baseline.accuracy * 100).toFixed(2)}%{' '}
                    <span className="text-[11px]">
                      [{(w750.metrics.baseline.accuracy_ci_95?.[0] ? w750.metrics.baseline.accuracy_ci_95[0] * 100 : 0).toFixed(1)}% -{' '}
                      {(w750.metrics.baseline.accuracy_ci_95?.[1] ? w750.metrics.baseline.accuracy_ci_95[1] * 100 : 0).toFixed(1)}%]
                    </span>
                  </td>
                </tr>
                <tr className="hover:bg-[#161B22]">
                  <td className="py-2 px-3 text-[#E6EDF3] font-semibold">Balanced Accuracy (95% CI)</td>
                  <td className="py-2 px-3">
                    {(w252.metrics.model.balanced_accuracy * 100).toFixed(2)}%{' '}
                    <span className="text-[11px] text-[#8B949E]">
                      [{(w252.metrics.model.balanced_accuracy_ci_95?.[0] ? w252.metrics.model.balanced_accuracy_ci_95[0] * 100 : 0).toFixed(1)}% -{' '}
                      {(w252.metrics.model.balanced_accuracy_ci_95?.[1] ? w252.metrics.model.balanced_accuracy_ci_95[1] * 100 : 0).toFixed(1)}%]
                    </span>
                  </td>
                  <td className="py-2 px-3">
                    {(w750.metrics.model.balanced_accuracy * 100).toFixed(2)}%{' '}
                    <span className="text-[11px] text-[#8B949E]">
                      [{(w750.metrics.model.balanced_accuracy_ci_95?.[0] ? w750.metrics.model.balanced_accuracy_ci_95[0] * 100 : 0).toFixed(1)}% -{' '}
                      {(w750.metrics.model.balanced_accuracy_ci_95?.[1] ? w750.metrics.model.balanced_accuracy_ci_95[1] * 100 : 0).toFixed(1)}%]
                    </span>
                  </td>
                </tr>
                <tr className="hover:bg-[#161B22]">
                  <td className="py-2 px-3 text-[#E6EDF3] font-semibold">Brier Score Loss (Lower is Better)</td>
                  <td className="py-2 px-3 text-[#3FB950] font-bold">
                    {w252.metrics.model.brier_score.toFixed(4)}{' '}
                    <span className="text-[#8B949E] font-normal">(Base: {w252.metrics.baseline.brier_score.toFixed(4)})</span>
                  </td>
                  <td className="py-2 px-3 text-[#3FB950] font-bold">
                    {w750.metrics.model.brier_score.toFixed(4)}{' '}
                    <span className="text-[#8B949E] font-normal">(Base: {w750.metrics.baseline.brier_score.toFixed(4)})</span>
                  </td>
                </tr>
                <tr className="hover:bg-[#161B22]">
                  <td className="py-2 px-3 text-[#E6EDF3] font-semibold">Log Loss (Lower is Better)</td>
                  <td className="py-2 px-3">
                    {w252.metrics.model.log_loss.toFixed(4)}{' '}
                    <span className="text-[#8B949E]">(Base: {w252.metrics.baseline.log_loss.toFixed(4)})</span>
                  </td>
                  <td className="py-2 px-3">
                    {w750.metrics.model.log_loss.toFixed(4)}{' '}
                    <span className="text-[#8B949E]">(Base: {w750.metrics.baseline.log_loss.toFixed(4)})</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {/* 3. SINGLE WINDOW FOCUS VIEW WITH STATISTICAL SCORECARD */}
        {(selectedView === '252' || selectedView === '750') && (
          <div className="mt-4 space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
                <div className="text-[11px] text-[#8B949E]">Directional Accuracy</div>
                <div className="text-xl font-bold font-mono tabular-nums text-[#E6EDF3] mt-1">
                  {(activeResult.metrics.model.accuracy * 100).toFixed(2)}%
                </div>
                <div className="text-[10px] text-[#8B949E] font-mono mt-0.5">
                  95% CI: [{(activeResult.metrics.model.accuracy_ci_95?.[0] ? activeResult.metrics.model.accuracy_ci_95[0] * 100 : 0).toFixed(1)}% -{' '}
                  {(activeResult.metrics.model.accuracy_ci_95?.[1] ? activeResult.metrics.model.accuracy_ci_95[1] * 100 : 0).toFixed(1)}%]
                </div>
              </div>

              <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
                <div className="text-[11px] text-[#8B949E]">Balanced Accuracy</div>
                <div className="text-xl font-bold font-mono tabular-nums text-[#58A6FF] mt-1">
                  {(activeResult.metrics.model.balanced_accuracy * 100).toFixed(2)}%
                </div>
                <div className="text-[10px] text-[#8B949E] font-mono mt-0.5">
                  95% CI: [{(activeResult.metrics.model.balanced_accuracy_ci_95?.[0] ? activeResult.metrics.model.balanced_accuracy_ci_95[0] * 100 : 0).toFixed(1)}% -{' '}
                  {(activeResult.metrics.model.balanced_accuracy_ci_95?.[1] ? activeResult.metrics.model.balanced_accuracy_ci_95[1] * 100 : 0).toFixed(1)}%]
                </div>
              </div>

              <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
                <div className="text-[11px] text-[#8B949E]">Brier Score Loss</div>
                <div className="text-xl font-bold font-mono tabular-nums text-[#3FB950] mt-1">
                  {activeResult.metrics.model.brier_score.toFixed(4)}
                </div>
                <div className="text-[10px] text-[#8B949E] font-mono mt-0.5">
                  Baseline: {activeResult.metrics.baseline.brier_score.toFixed(4)}
                </div>
              </div>

              <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded">
                <div className="text-[11px] text-[#8B949E]">Log Loss</div>
                <div className="text-xl font-bold font-mono tabular-nums text-[#E6EDF3] mt-1">
                  {activeResult.metrics.model.log_loss.toFixed(4)}
                </div>
                <div className="text-[10px] text-[#8B949E] font-mono mt-0.5">
                  Baseline: {activeResult.metrics.baseline.log_loss.toFixed(4)}
                </div>
              </div>
            </div>

            {/* Model vs Baseline Audit Scorecard (Audit Requirement 8) */}
            <div className="p-4 bg-[#090C10] border border-[#1E2638] rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-[#58A6FF]" />
                  <span className="text-xs font-semibold text-[#E6EDF3]">
                    Out-of-Sample Hypothesis Test vs Baseline Scorecard
                  </span>
                </div>
                <span className="text-xs font-mono text-[#8B949E]">
                  Sample: {activeResult.evaluated_test_sessions} sessions · McNemar p={activeResult.metrics.edge.p_value_mcnemar}
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono tabular-nums mt-3">
                <div className="p-2 bg-[#121721] rounded border border-[#1E2638]">
                  <span className="text-[10px] text-[#8B949E]">Directional Acc Edge</span>
                  <div className={`font-bold ${activeResult.metrics.edge.beats_accuracy ? 'text-[#3FB950]' : 'text-[#F85149]'}`}>
                    {activeResult.metrics.edge.accuracy_diff > 0 ? '+' : ''}
                    {(activeResult.metrics.edge.accuracy_diff * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="p-2 bg-[#121721] rounded border border-[#1E2638]">
                  <span className="text-[10px] text-[#8B949E]">Balanced Acc Edge</span>
                  <div className={`font-bold ${activeResult.metrics.edge.beats_balanced_accuracy ? 'text-[#3FB950]' : 'text-[#F85149]'}`}>
                    {activeResult.metrics.edge.balanced_acc_diff > 0 ? '+' : ''}
                    {(activeResult.metrics.edge.balanced_acc_diff * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="p-2 bg-[#121721] rounded border border-[#1E2638]">
                  <span className="text-[10px] text-[#8B949E]">Brier Improvement</span>
                  <div className={`font-bold ${activeResult.metrics.edge.beats_brier ? 'text-[#3FB950]' : 'text-[#F85149]'}`}>
                    {activeResult.metrics.edge.brier_diff > 0 ? '+' : ''}
                    {activeResult.metrics.edge.brier_diff.toFixed(4)}
                  </div>
                </div>

                <div className="p-2 bg-[#121721] rounded border border-[#1E2638]">
                  <span className="text-[10px] text-[#8B949E]">LogLoss Improvement</span>
                  <div className={`font-bold ${activeResult.metrics.edge.beats_logloss ? 'text-[#3FB950]' : 'text-[#F85149]'}`}>
                    {activeResult.metrics.edge.logloss_diff > 0 ? '+' : ''}
                    {activeResult.metrics.edge.logloss_diff.toFixed(4)}
                  </div>
                </div>
              </div>

              <div className="mt-3 text-xs text-[#8B949E] leading-relaxed pt-2 border-t border-[#1E2638]">
                <span className="text-[#E6EDF3] font-semibold">Statistical Audit Conclusion: </span>
                {activeResult.metrics.edge.summary_conclusion}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Calibration Curve & Feature Weights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reliability Calibration Diagram */}
        <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h4 className="text-xs font-semibold text-[#E6EDF3]">Probability Calibration (Reliability Diagram)</h4>
              <p className="text-[11px] text-[#8B949E]">
                Comparing predicted P(Up) bin against realized fraction of Up sessions
              </p>
            </div>
            <div className="text-xs text-[#8B949E] font-mono">
              Window {selectedView === '750' ? '750' : '252'}
            </div>
          </div>

          <div className="mt-4">
            <svg viewBox="0 0 400 240" className="w-full h-56 select-none">
              {/* 45 degree perfect calibration line */}
              <line x1="40" y1="200" x2="360" y2="40" stroke="#30363D" strokeWidth="1.5" strokeDasharray="4 4" />
              <text x="365" y="45" fill="#8B949E" fontSize="9" fontFamily="monospace">
                Ideal (x=y)
              </text>

              {/* Grid Lines */}
              {[0.2, 0.4, 0.6, 0.8].map((v, i) => {
                const x = 40 + v * 320;
                const y = 200 - v * 160;
                return (
                  <g key={i}>
                    <line x1="40" y1={y} x2="360" y2={y} stroke="#1E2638" strokeDasharray="2 2" />
                    <text x="32" y={y + 3} fill="#8B949E" fontSize="8" fontFamily="monospace" textAnchor="end">
                      {(v * 100).toFixed(0)}%
                    </text>
                    <text x={x} y="215" fill="#8B949E" fontSize="8" fontFamily="monospace" textAnchor="middle">
                      {(v * 100).toFixed(0)}%
                    </text>
                  </g>
                );
              })}

              {/* Calibration Points with CI bars */}
              {activeResult.calibration.map((bin, i) => {
                const cx = 40 + bin.mean_predicted_prob * 320;
                const cy = 200 - bin.empirical_realized_rate * 160;
                const isSmall = bin.is_small_sample;

                return (
                  <g key={i}>
                    <circle
                      cx={cx}
                      cy={cy}
                      r="5"
                      fill={isSmall ? '#D29922' : '#58A6FF'}
                      stroke="#090C10"
                      strokeWidth="2"
                    />
                    <text
                      x={cx}
                      y={cy - 8}
                      fill={isSmall ? '#D29922' : '#E6EDF3'}
                      fontSize="8"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      n={bin.sample_count}
                    </text>
                  </g>
                );
              })}

              <text x="200" y="235" fill="#8B949E" fontSize="10" fontFamily="monospace" textAnchor="middle">
                Mean Predicted P(Up)
              </text>
            </svg>
          </div>

          {/* Calibration Table with Sample Sizes and Confidence Intervals (Requirement 7) */}
          <div className="mt-3 pt-3 border-t border-[#1E2638] overflow-x-auto">
            <table className="w-full text-left text-[11px] font-mono tabular-nums">
              <thead>
                <tr className="text-[#8B949E] border-b border-[#1E2638]">
                  <th className="py-1">Bin Range</th>
                  <th className="py-1">Count</th>
                  <th className="py-1">Mean P(Up)</th>
                  <th className="py-1">Empirical Rate (95% CI)</th>
                  <th className="py-1 text-right">Sample Reliability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E2638]/50">
                {activeResult.calibration.map((bin, idx) => (
                  <tr key={idx}>
                    <td className="py-1 text-[#E6EDF3]">{bin.range}</td>
                    <td className="py-1">{bin.sample_count}</td>
                    <td className="py-1">{(bin.mean_predicted_prob * 100).toFixed(1)}%</td>
                    <td className="py-1">
                      {(bin.empirical_realized_rate * 100).toFixed(1)}%{' '}
                      {bin.calibration_ci_95 && (
                        <span className="text-[#8B949E]">
                          [{(bin.calibration_ci_95[0] * 100).toFixed(0)}%-{(bin.calibration_ci_95[1] * 100).toFixed(0)}%]
                        </span>
                      )}
                    </td>
                    <td className="py-1 text-right">
                      {bin.is_small_sample ? (
                        <span className="text-[#D29922] font-semibold">Small (n &lt; 30)</span>
                      ) : (
                        <span className="text-[#3FB950]">Adequate (n &ge; 30)</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-2 text-[10px] text-[#8B949E]">
            Caution: Bins with sample size n &lt; 30 exhibit high estimation variance. Avoid drawing strong conclusions from sparsely populated extreme probability buckets.
          </div>
        </div>

        {/* Standardized Feature Coefficients */}
        <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
          <h4 className="text-xs font-semibold text-[#E6EDF3] mb-1">Standardized Model Feature Coefficients</h4>
          <p className="text-[11px] text-[#8B949E] mb-3 font-mono">
            Fitted strictly on training data with frozen StandardScaler (zero test leakage)
          </p>

          <div className="space-y-2 mt-2">
            {activeResult.feature_importances.map((item, idx) => {
              const isBull = item.coefficient >= 0;
              const maxCoeff = Math.max(...activeResult.feature_importances.map((f) => f.abs_importance), 0.5);
              const barWidth = `${(item.abs_importance / maxCoeff) * 100}%`;

              return (
                <div key={idx} className="p-2 bg-[#090C10] border border-[#1E2638] rounded text-xs font-mono tabular-nums">
                  <div className="flex items-center justify-between">
                    <span className="text-[#E6EDF3] font-semibold">{item.feature}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-[11px] text-[#8B949E]">OR: {item.odds_ratio}</span>
                      <span className={`font-bold ${isBull ? 'text-[#3FB950]' : 'text-[#F85149]'}`}>
                        {item.coefficient > 0 ? '+' : ''}
                        {item.coefficient.toFixed(4)}
                      </span>
                    </div>
                  </div>
                  <div className="w-full bg-[#1E2638] h-1 rounded-full mt-1.5 overflow-hidden">
                    <div
                      className={`h-full ${isBull ? 'bg-[#3FB950]' : 'bg-[#F85149]'}`}
                      style={{ width: barWidth }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
