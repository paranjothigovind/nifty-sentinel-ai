import React, { useState } from 'react';
import { MarketDataStatusSummary, DataMode } from '../types';
import { Activity, ShieldAlert, Clock, Database, CheckCircle, AlertTriangle, XCircle, RefreshCw } from 'lucide-react';

interface DataStatusPanelProps {
  status: MarketDataStatusSummary | null;
  onRefresh?: () => void;
  onSwitchMode?: (mode: string) => void;
  isSwitching?: boolean;
}

export const DataStatusPanel: React.FC<DataStatusPanelProps> = ({
  status,
  onRefresh,
  onSwitchMode,
  isSwitching = false,
}) => {
  const [selectedMode, setSelectedMode] = useState<string>(status?.active_mode || 'replay');

  if (!status) {
    return (
      <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg animate-pulse text-xs text-[#8B949E]">
        Loading intraday market data status...
      </div>
    );
  }

  const modeBadgeColor = (mode: DataMode) => {
    switch (mode) {
      case 'LIVE':
        return 'bg-[#238636]/20 border-[#3FB950] text-[#3FB950]';
      case 'DELAYED':
        return 'bg-[#D29922]/20 border-[#D29922] text-[#E3B341]';
      case 'REPLAY':
        return 'bg-[#A371F7]/20 border-[#BC8CFF] text-[#D2A8FF]';
      case 'HISTORICAL':
        return 'bg-[#1F6FEB]/20 border-[#58A6FF] text-[#79C0FF]';
      case 'UNAVAILABLE':
      default:
        return 'bg-[#F85149]/20 border-[#F85149] text-[#FFA198]';
    }
  };

  const handleModeChange = (mode: string) => {
    setSelectedMode(mode);
    if (onSwitchMode) {
      onSwitchMode(mode);
    }
  };

  return (
    <div className="bg-[#121721] border border-[#1E2638] rounded-xl p-4 sm:p-5 space-y-4">
      {/* Header row: Title, Prominent Data Mode Badge, and Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1E2638] pb-3">
        <div className="flex items-center gap-2.5">
          <Activity className="w-4 h-4 text-[#58A6FF]" />
          <span className="text-sm font-semibold text-[#E6EDF3]">
            Intraday Market Data Layer
          </span>
          {/* Prominent Data Mode Badge */}
          <span
            className={`px-2.5 py-0.5 text-[11px] font-mono font-bold uppercase rounded border ${modeBadgeColor(
              status.data_mode
            )}`}
          >
            MODE: {status.data_mode}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Mode Switcher */}
          <div className="flex items-center bg-[#090C10] border border-[#1E2638] rounded-md p-0.5 text-[11px] font-mono">
            <button
              onClick={() => handleModeChange('replay')}
              disabled={isSwitching}
              className={`px-2.5 py-1 rounded transition-colors ${
                status.active_mode === 'replay'
                  ? 'bg-[#1E2638] text-[#E6EDF3] font-bold'
                  : 'text-[#8B949E] hover:text-[#E6EDF3]'
              }`}
            >
              CSV Replay
            </button>
            <button
              onClick={() => handleModeChange('free_index')}
              disabled={isSwitching}
              className={`px-2.5 py-1 rounded transition-colors ${
                status.active_mode === 'free_index'
                  ? 'bg-[#1E2638] text-[#E6EDF3] font-bold'
                  : 'text-[#8B949E] hover:text-[#E6EDF3]'
              }`}
            >
              Yahoo Feed (15m Delay)
            </button>
          </div>

          {onRefresh && (
            <button
              onClick={onRefresh}
              className="p-1.5 text-[#8B949E] hover:text-[#E6EDF3] hover:bg-[#1E2638] rounded transition-colors"
              title="Refresh Market Data Status"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isSwitching ? 'animate-spin' : ''}`} />
            </button>
          )}
        </div>
      </div>

      {/* Grid of Key Provider and Freshness Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
        {/* Metric 1: Index Provider & Last Market Bar */}
        <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded-lg space-y-1">
          <div className="text-[11px] text-[#8B949E] flex items-center justify-between">
            <span>Index Feed (5m Bars)</span>
            <span
              className={`w-2 h-2 rounded-full ${
                status.index_provider.healthy ? 'bg-[#3FB950]' : 'bg-[#F85149]'
              }`}
            />
          </div>
          <div className="font-medium text-[#E6EDF3] truncate" title={status.index_provider.name}>
            {status.index_provider.id === 'csv_replay' ? 'Deterministic CSV Replay' : 'Yahoo ^NSEI Public API'}
          </div>
          <div className="text-[11px] text-[#8B949E] font-mono">
            {status.index_provider.last_market_ts_ist || 'No candles'}
          </div>
        </div>

        {/* Metric 2: Options Chain Provider & Coverage */}
        <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded-lg space-y-1">
          <div className="text-[11px] text-[#8B949E] flex items-center justify-between">
            <span>Options Provider</span>
            <span
              className={`w-2 h-2 rounded-full ${
                status.options_provider.healthy ? 'bg-[#3FB950]' : 'bg-[#D29922]'
              }`}
            />
          </div>
          <div className="font-medium text-[#E6EDF3] truncate" title={status.options_provider.name}>
            {status.options_provider.id === 'csv_replay'
              ? 'Replay Options Chain'
              : 'Unavailable (No Broker Feed)'}
          </div>
          <div className="text-[11px] text-[#8B949E] font-mono">
            Contracts: {status.options_provider.contracts_count} strikes loaded
          </div>
        </div>

        {/* Metric 3: Quote Age & Freshness Safeguards */}
        <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded-lg space-y-1">
          <div className="text-[11px] text-[#8B949E] flex items-center gap-1">
            <Clock className="w-3 h-3" />
            <span>Freshness Safeguards</span>
          </div>
          <div className="font-mono text-[#E6EDF3]">
            {status.active_mode === 'replay' ? (
              <span className="text-[#BC8CFF]">Replay Valid (Static)</span>
            ) : status.index_provider.age_seconds > 60 ? (
              <span className="text-[#F85149] font-bold">
                {status.index_provider.age_seconds}s old (Stale)
              </span>
            ) : (
              <span className="text-[#3FB950]">
                {status.index_provider.age_seconds}s old (&lt; 60s)
              </span>
            )}
          </div>
          <div className="text-[11px] text-[#8B949E]">
            Bar req: Completed 5m · Max age: 15s
          </div>
        </div>

        {/* Metric 4: Market Session & Hours */}
        <div className="p-3 bg-[#090C10] border border-[#1E2638] rounded-lg space-y-1">
          <div className="text-[11px] text-[#8B949E] flex items-center justify-between">
            <span>NSE Session Status</span>
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                status.market_session.is_open
                  ? 'bg-[#238636]/30 text-[#3FB950]'
                  : 'bg-[#1E2638] text-[#8B949E]'
              }`}
            >
              {status.market_session.is_open ? 'OPEN' : 'CLOSED'}
            </span>
          </div>
          <div className="text-[#E6EDF3] text-[11px] truncate" title={status.market_session.session_note}>
            {status.market_session.session_note}
          </div>
          <div className="text-[10px] text-[#8B949E] font-mono">
            Hours: 09:15 - 15:30 IST
          </div>
        </div>
      </div>

      {/* Signal Gating & Trade Block Reasons Banner */}
      <div
        className={`p-3.5 rounded-lg border text-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 ${
          status.is_paper_trading_ready
            ? 'bg-[#161B22] border-[#238636]/40 text-[#7EE787]'
            : 'bg-[#1F1315] border-[#F85149]/30 text-[#FFA198]'
        }`}
      >
        <div className="flex items-start gap-2.5">
          {status.is_paper_trading_ready ? (
            <CheckCircle className="w-4 h-4 text-[#3FB950] shrink-0 mt-0.5" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-[#F85149] shrink-0 mt-0.5" />
          )}
          <div>
            <div className="font-semibold text-[#E6EDF3]">
              {status.is_paper_trading_ready
                ? 'Paper Trading Ready (Deterministic Replay Mode)'
                : 'Intraday CALL/PUT Paper Entry Blocked'}
            </div>
            {status.trade_signal_blocked_reasons.length > 0 ? (
              <div className="mt-1 text-[11px] text-[#FFA198] space-y-0.5 font-mono">
                {status.trade_signal_blocked_reasons.map((reason, idx) => (
                  <div key={idx} className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#F85149]" />
                    <span>{reason}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="mt-0.5 text-[11px] text-[#8B949E]">
                All data integrity and completed-bar freshness criteria satisfied.
              </div>
            )}
          </div>
        </div>

        {/* Safety Badge */}
        <div className="flex items-center gap-1.5 text-[11px] font-mono text-[#8B949E] bg-[#090C10] px-2.5 py-1 rounded border border-[#1E2638] shrink-0">
          <ShieldAlert className="w-3.5 h-3.5 text-[#58A6FF]" />
          <span>Zero Live Broker Execution</span>
        </div>
      </div>
    </div>
  );
};
