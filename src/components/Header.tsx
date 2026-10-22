import React from 'react';
import { RefreshCw, Terminal, Cpu } from 'lucide-react';

interface HeaderProps {
  activeTab: string;
  onSelectTab: (tab: string) => void;
  onSync: () => void;
  isSyncing: boolean;
  onOpenAudit: () => void;
  pythonStatus: 'healthy' | 'checking' | 'error';
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onSelectTab,
  onSync,
  isSyncing,
  onOpenAudit,
  pythonStatus,
}) => {
  return (
    <header className="sticky top-0 z-40 flex items-center justify-between gap-8 px-6 py-3.5 bg-[#090C10] border-b border-[#1E2638]">
      {/* Zone 1: Single text element wordmark */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded bg-[#161B22] border border-[#30363D] flex items-center justify-center text-[#58A6FF]">
          <Terminal className="w-4 h-4" />
        </div>
        <div className="flex flex-col">
          <span className="text-base font-bold tracking-tight text-[#E6EDF3] whitespace-nowrap">
            NIFTY Sentinel AI
          </span>
          <span className="text-[11px] text-[#8B949E] font-mono tabular-nums leading-none">
            NSE: ^NSEI · Quant Lab Phase 1
          </span>
        </div>
      </div>

      {/* Zone 2: 4-5 single-line clean nav links */}
      <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-[#8B949E]">
        <button
          onClick={() => onSelectTab('overview')}
          className={`hover:text-[#E6EDF3] transition-colors whitespace-nowrap pb-1 ${
            activeTab === 'overview' ? 'text-[#58A6FF] border-b-2 border-[#58A6FF]' : ''
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => onSelectTab('features')}
          className={`hover:text-[#E6EDF3] transition-colors whitespace-nowrap pb-1 ${
            activeTab === 'features' ? 'text-[#58A6FF] border-b-2 border-[#58A6FF]' : ''
          }`}
        >
          Features & Target
        </button>
        <button
          onClick={() => onSelectTab('walk-forward')}
          className={`hover:text-[#E6EDF3] transition-colors whitespace-nowrap pb-1 ${
            activeTab === 'walk-forward' ? 'text-[#58A6FF] border-b-2 border-[#58A6FF]' : ''
          }`}
        >
          Walk-Forward
        </button>
        <button
          onClick={() => onSelectTab('predictions')}
          className={`hover:text-[#E6EDF3] transition-colors whitespace-nowrap pb-1 ${
            activeTab === 'predictions' ? 'text-[#58A6FF] border-b-2 border-[#58A6FF]' : ''
          }`}
        >
          Prediction Ledger
        </button>
        <button
          onClick={onOpenAudit}
          className="hover:text-[#E6EDF3] transition-colors whitespace-nowrap pb-1 flex items-center gap-1.5"
        >
          <Cpu className="w-3.5 h-3.5 text-[#8B949E]" />
          System Manifest
        </button>
      </nav>

      {/* Zone 3: 1 primary action */}
      <div className="flex items-center gap-3 shrink-0">
        <button
          onClick={onSync}
          disabled={isSyncing}
          className="flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium text-white bg-[#238636] hover:bg-[#2ea043] disabled:opacity-50 transition-colors rounded whitespace-nowrap cursor-pointer shadow-sm"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isSyncing ? 'animate-spin' : ''}`} />
          <span>{isSyncing ? 'Ingesting OHLCV...' : 'Sync Market Data'}</span>
        </button>
      </div>
    </header>
  );
};
