import React from 'react';
import { SystemStatus, ManifestItem } from '../types';
import { X, Cpu, CheckCircle, Database, Shield, ExternalLink } from 'lucide-react';

interface SystemAuditDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  status: SystemStatus | null;
  manifest: ManifestItem[];
}

export const SystemAuditDrawer: React.FC<SystemAuditDrawerProps> = ({
  isOpen,
  onClose,
  status,
  manifest,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-xs">
      <div className="w-full max-w-2xl h-full bg-[#090C10] border-l border-[#1E2638] p-6 overflow-y-auto space-y-6">
        <div className="flex items-center justify-between pb-4 border-b border-[#1E2638]">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-[#58A6FF]" />
            <h2 className="text-base font-bold text-[#E6EDF3]">System Architecture & Component Manifest</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-[#8B949E] hover:text-[#E6EDF3] transition-colors rounded hover:bg-[#1E2638]"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Runtime Environment Status */}
        {status && (
          <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg space-y-3">
            <h3 className="text-xs font-semibold text-[#58A6FF] uppercase tracking-wider">
              Active Runtime Environment
            </h3>
            <div className="grid grid-cols-2 gap-3 text-xs font-mono tabular-nums">
              <div>
                <span className="text-[#8B949E]">Python Runtime:</span>
                <div className="text-[#E6EDF3] mt-0.5">{status.runtime.python_version} ({status.runtime.platform})</div>
              </div>
              <div>
                <span className="text-[#8B949E]">FastAPI Microservice:</span>
                <div className="text-[#3FB950] mt-0.5">RUNNING on 127.0.0.1:{status.runtime.fastapi_port}</div>
              </div>
              <div>
                <span className="text-[#8B949E]">Express Public Gateway:</span>
                <div className="text-[#3FB950] mt-0.5">LISTENING on 0.0.0.0:3000</div>
              </div>
              <div>
                <span className="text-[#8B949E]">SQLite Storage:</span>
                <div className="text-[#E6EDF3] mt-0.5">
                  {status.database_counts.nifty_daily_ohlcv} sessions stored
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-[#1E2638] text-xs font-mono">
              <span className="text-[#8B949E]">Verified Packages:</span>{' '}
              <span className="text-[#E6EDF3]">
                fastapi {status.packages.fastapi} · scikit-learn {status.packages.scikit_learn} · pandas {status.packages.pandas} · numpy {status.packages.numpy} · yfinance {status.packages.yfinance}
              </span>
            </div>
          </div>
        )}

        {/* Component Manifest Table */}
        <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg">
          <h3 className="text-xs font-semibold text-[#58A6FF] uppercase tracking-wider mb-2">
            System Component Manifest (Verification Matrix)
          </h3>
          <p className="text-xs text-[#8B949E] mb-4">
            Auditing which parts of the quantitative pipeline are live, which are test-only fixtures, and which depend on external providers.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
              <thead>
                <tr className="border-b border-[#1E2638] text-[#8B949E] bg-[#090C10]">
                  <th className="py-2 px-2.5">Component</th>
                  <th className="py-2 px-2.5">Status</th>
                  <th className="py-2 px-2.5">Source / Execution</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E2638]/60">
                {manifest.map((item, idx) => (
                  <tr key={idx} className="hover:bg-[#161B22]">
                    <td className="py-2.5 px-2.5">
                      <div className="font-semibold text-[#E6EDF3]">{item.name}</div>
                      <div className="text-[11px] text-[#8B949E] font-sans">{item.notes}</div>
                    </td>
                    <td className="py-2.5 px-2.5 whitespace-nowrap">
                      {item.status === 'LIVE' ? (
                        <span className="text-[#3FB950] font-bold flex items-center gap-1">
                          <CheckCircle className="w-3.5 h-3.5" /> LIVE
                        </span>
                      ) : item.status === 'TEST_ONLY' ? (
                        <span className="text-[#D29922] font-bold flex items-center gap-1">
                          <Shield className="w-3.5 h-3.5" /> TEST FIXTURE
                        </span>
                      ) : (
                        <span className="text-[#58A6FF] font-bold flex items-center gap-1">
                          <ExternalLink className="w-3.5 h-3.5" /> EXTERNAL
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-2.5 text-[#8B949E] text-[11px] truncate max-w-xs">
                      {item.source}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Academic Quant Research Rules */}
        <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg text-xs space-y-2 text-[#8B949E]">
          <h4 className="font-semibold text-[#E6EDF3]">Quantitative Integrity Guarantee</h4>
          <ul className="list-disc list-inside space-y-1">
            <li>Zero simulated metrics or fabricated forecasts are ever rendered in the live UI.</li>
            <li>Synthetic data is restricted exclusively to the automated unit test suite.</li>
            <li>Class distribution is dynamically calculated from actual market sessions.</li>
            <li>Automated trading is strictly prohibited in Phase 1.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
