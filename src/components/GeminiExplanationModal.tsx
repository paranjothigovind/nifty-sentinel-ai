import React from 'react';
import { X, Sparkles, AlertCircle, BookOpen } from 'lucide-react';

interface GeminiExplanationModalProps {
  isOpen: boolean;
  onClose: () => void;
  explanation: string | null;
  isLoading: boolean;
  error?: string | null;
}

export const GeminiExplanationModal: React.FC<GeminiExplanationModalProps> = ({
  isOpen,
  onClose,
  explanation,
  isLoading,
  error,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs">
      <div className="w-full max-w-2xl bg-[#090C10] border border-[#1E2638] rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#1E2638] bg-[#121721]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded bg-[#161B22] border border-[#30363D] flex items-center justify-center text-[#58A6FF]">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-[#E6EDF3]">Quantitative Regime Analysis (Gemini AI Layer)</h3>
              <p className="text-[11px] text-[#8B949E]">
                Post-hoc statistical interpretation of verified walk-forward metrics & feature weights
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#8B949E] hover:text-[#E6EDF3] transition-colors p-1 rounded hover:bg-[#1E2638]"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-4 text-xs text-[#E6EDF3] leading-relaxed">
          {isLoading ? (
            <div className="py-12 text-center space-y-3">
              <Sparkles className="w-8 h-8 text-[#58A6FF] animate-spin mx-auto opacity-70" />
              <p className="text-sm font-medium text-[#E6EDF3]">
                Synthesizing Quantitative Metrics...
              </p>
              <p className="text-xs text-[#8B949E]">
                Querying server-side Gemini 3.8 Flash with verified empirical results
              </p>
            </div>
          ) : error ? (
            <div className="p-4 bg-[#1F1315] border border-[#F85149]/30 rounded-lg flex items-start gap-3 text-[#F85149]">
              <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold text-sm">Explanation Generation Failed</div>
                <p className="text-xs mt-1 text-[#F85149]/90">{error}</p>
              </div>
            </div>
          ) : explanation ? (
            <div className="space-y-4">
              <div className="p-3 bg-[#121721] border border-[#1E2638] rounded flex items-center gap-2 text-[#8B949E] text-[11px]">
                <BookOpen className="w-4 h-4 text-[#58A6FF] shrink-0" />
                <span>
                  This analysis is strictly an educational explanation layer. Gemini does not generate or alter the quantitative model weights or numerical predictions.
                </span>
              </div>

              <div className="prose prose-invert max-w-none text-xs leading-relaxed text-[#E6EDF3] space-y-3 whitespace-pre-line font-sans">
                {explanation}
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-[#8B949E]">
              No explanation available. Trigger analysis from the Prediction Ledger.
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-[#1E2638] bg-[#121721] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium text-[#E6EDF3] bg-[#1E2638] hover:bg-[#28334b] border border-[#30363D] rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
