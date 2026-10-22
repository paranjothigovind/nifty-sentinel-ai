import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { DataQualityCard } from './components/DataQualityCard';
import { PriceChart } from './components/PriceChart';
import { FeatureInspector } from './components/FeatureInspector';
import { WalkForwardMatrix } from './components/WalkForwardMatrix';
import { PredictionLedger } from './components/PredictionLedger';
import { SystemAuditDrawer } from './components/SystemAuditDrawer';
import { GeminiExplanationModal } from './components/GeminiExplanationModal';
import { DataStatusPanel } from './components/DataStatusPanel';
import {
  OHLCVRecord,
  DataQualityAudit,
  FeatureStats,
  FeatureRow,
  WindowComparison,
  PredictionRecord,
  SystemStatus,
  ManifestItem,
  MarketDataStatusSummary,
} from './types';
import { AlertCircle, Terminal, TrendingUp, ShieldAlert } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  const [isBacktesting, setIsBacktesting] = useState<boolean>(false);
  const [isExplaining, setIsExplaining] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Modals & Drawers
  const [isAuditOpen, setIsAuditOpen] = useState<boolean>(false);
  const [isExplanationOpen, setIsExplanationOpen] = useState<boolean>(false);
  const [explanationText, setExplanationText] = useState<string | null>(null);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  // Quantitative Application State
  const [historyData, setHistoryData] = useState<OHLCVRecord[]>([]);
  const [qualityAudit, setQualityAudit] = useState<DataQualityAudit | null>(null);
  const [featureStats, setFeatureStats] = useState<FeatureStats | null>(null);
  const [sampleFeatures, setSampleFeatures] = useState<FeatureRow[]>([]);
  const [backtestComparison, setBacktestComparison] = useState<WindowComparison | null>(null);
  const [latestPrediction, setLatestPrediction] = useState<PredictionRecord | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictionRecord[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [systemManifest, setSystemManifest] = useState<ManifestItem[]>([]);
  const [marketDataStatus, setMarketDataStatus] = useState<MarketDataStatusSummary | null>(null);
  const [isSwitchingDataMode, setIsSwitchingDataMode] = useState<boolean>(false);

  // Load all initial data from quantitative API
  const loadDashboardData = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      // 1. Fetch system status & manifest
      const statusRes = await fetch('/api/quant/system-status');
      if (statusRes.ok) {
        const sData = await statusRes.json();
        setSystemStatus(sData);
        if (sData.data_quality_audit) {
          setQualityAudit(sData.data_quality_audit);
        }
      }

      const manifestRes = await fetch('/api/quant/manifest');
      if (manifestRes.ok) {
        const mData = await manifestRes.json();
        setSystemManifest(mData.manifest || []);
      }

      // 1b. Fetch Intraday Market Data Layer status
      try {
        const mktStatusRes = await fetch('/api/market-data/status');
        if (mktStatusRes.ok) {
          const mktData = await mktStatusRes.json();
          setMarketDataStatus(mktData);
        }
      } catch (e) {
        console.warn('Could not fetch market data status:', e);
      }

      // 2. Fetch historical prices
      const histRes = await fetch('/api/quant/history');
      if (histRes.ok) {
        const hData = await histRes.json();
        setHistoryData(hData.data || []);
      }

      // 3. Fetch features & empirical class balance
      const featRes = await fetch('/api/quant/features');
      if (featRes.ok) {
        const fData = await featRes.json();
        setFeatureStats(fData.stats || null);
        setSampleFeatures(fData.features || []);
      }

      // 4. Fetch walk-forward 252 vs 750 comparison
      const compRes = await fetch('/api/quant/compare');
      if (compRes.ok) {
        const cData = await compRes.json();
        setBacktestComparison(cData);
      }

      // 5. Fetch predictions
      const predRes = await fetch('/api/quant/prediction/latest');
      if (predRes.ok) {
        const pData = await predRes.json();
        setLatestPrediction(pData);
      }

      const ledgerRes = await fetch('/api/quant/predictions?limit=50');
      if (ledgerRes.ok) {
        const lData = await ledgerRes.json();
        setPredictionHistory(lData.predictions || []);
      }
    } catch (err: any) {
      console.error('Failed to load quantitative data:', err);
      setErrorMessage('Failed to connect to quantitative backend. Please check system status.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Sync Market Data Handler
  const handleSyncData = async () => {
    setIsSyncing(true);
    setErrorMessage(null);
    try {
      const res = await fetch('/api/quant/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: '^NSEI', years: 5 }),
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || 'Data synchronization failed');
      }
      // Reload dashboard after sync
      await loadDashboardData();
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsSyncing(false);
    }
  };

  // Run specific backtest configuration
  const handleRunBacktest = async (minTrain: number) => {
    setIsBacktesting(true);
    setErrorMessage(null);
    try {
      const res = await fetch('/api/quant/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ min_train_size: minTrain, refit_interval: 60 }),
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || 'Backtest execution failed');
      }
      // Re-fetch comparison
      const compRes = await fetch('/api/quant/compare?refit_interval=60');
      if (compRes.ok) {
        setBacktestComparison(await compRes.json());
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsBacktesting(false);
    }
  };

  // Re-compute latest prediction
  const handleRefreshPrediction = async () => {
    try {
      const predRes = await fetch('/api/quant/prediction/latest');
      if (predRes.ok) {
        setLatestPrediction(await predRes.json());
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    }
  };

  // Post-hoc Gemini explanation
  const handleExplainWithGemini = async () => {
    setIsExplaining(true);
    setExplanationError(null);
    setExplanationText(null);
    setIsExplanationOpen(true);

    try {
      const res = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          backtest: backtestComparison?.window_252,
          prediction: latestPrediction,
          stats: featureStats,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.details || errData.error || 'Gemini API call failed');
      }

      const data = await res.json();
      setExplanationText(data.explanation);
    } catch (err: any) {
      setExplanationError(err.message);
    } finally {
      setIsExplaining(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#090C10] text-[#E6EDF3] flex flex-col font-sans selection:bg-[#58A6FF]/30">
      {/* 3-Zone Top Bar */}
      <Header
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onSync={handleSyncData}
        isSyncing={isSyncing}
        onOpenAudit={() => setIsAuditOpen(true)}
        pythonStatus={systemStatus ? 'healthy' : 'checking'}
      />

      {/* Main Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
        {/* Error notification banner */}
        {errorMessage && (
          <div className="p-4 bg-[#1F1315] border border-[#F85149]/30 rounded-lg flex items-center justify-between text-[#F85149] text-xs">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-[#8B949E] hover:text-[#E6EDF3] font-mono"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Global Loading state */}
        {isLoading && historyData.length === 0 ? (
          <div className="py-24 text-center space-y-3">
            <div className="w-8 h-8 border-2 border-[#58A6FF] border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-sm font-medium text-[#E6EDF3]">Initializing Quantitative Engine...</p>
            <p className="text-xs text-[#8B949E]">
              Connecting to Python FastAPI service on 127.0.0.1:8050
            </p>
          </div>
        ) : (
          <>
            {/* VIEW 1: Overview & Price Candlestick */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                <DataQualityCard
                  audit={qualityAudit}
                  totalRecords={historyData.length}
                  latestSourceTs={historyData[historyData.length - 1]?.source_ts}
                  dataSource={historyData[0]?.data_source || 'yfinance'}
                />

                <PriceChart data={historyData} />

                {/* Key Quantitative Metrics Strip */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg">
                    <div className="text-[11px] text-[#8B949E]">Historical Trading Sessions</div>
                    <div className="text-xl font-bold font-mono tabular-nums text-[#E6EDF3] mt-1">
                      {historyData.length} Days
                    </div>
                    <div className="text-[11px] text-[#8B949E] mt-0.5 font-mono">
                      ~5 Years (Daily OHLCV)
                    </div>
                  </div>

                  <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg">
                    <div className="text-[11px] text-[#8B949E]">Empirical Up Ratio</div>
                    <div className="text-xl font-bold font-mono tabular-nums text-[#3FB950] mt-1">
                      {featureStats?.up_percentage ? `${featureStats.up_percentage.toFixed(2)}%` : '—'}
                    </div>
                    <div className="text-[11px] text-[#8B949E] mt-0.5 font-mono">
                      {featureStats?.up_count} Up / {featureStats?.down_count} Down
                    </div>
                  </div>

                  <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg">
                    <div className="text-[11px] text-[#8B949E]">Walk-Forward Out-of-Sample Hit Rate</div>
                    <div className="text-xl font-bold font-mono tabular-nums text-[#58A6FF] mt-1">
                      {backtestComparison?.window_252?.metrics?.model?.accuracy
                        ? `${(backtestComparison.window_252.metrics.model.accuracy * 100).toFixed(2)}%`
                        : '—'}
                    </div>
                    <div className="text-[11px] text-[#8B949E] mt-0.5 font-mono">
                      Base: {backtestComparison?.window_252?.metrics?.baseline?.accuracy
                        ? `${(backtestComparison.window_252.metrics.baseline.accuracy * 100).toFixed(2)}%`
                        : '—'}
                    </div>
                  </div>

                  <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg">
                    <div className="text-[11px] text-[#8B949E]">Latest Probability Forecast</div>
                    <div className="text-xl font-bold font-mono tabular-nums text-[#E6EDF3] mt-1">
                      {latestPrediction?.p_up ? `${(latestPrediction.p_up * 100).toFixed(1)}% Up` : '—'}
                    </div>
                    <div className="text-[11px] text-[#8B949E] mt-0.5 font-mono">
                      Target: {latestPrediction?.target_date || 'Pending'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* VIEW 2: Features & Dynamic Class Balance */}
            {activeTab === 'features' && (
              <FeatureInspector stats={featureStats} features={sampleFeatures} />
            )}

            {/* VIEW 3: Walk-Forward 252 vs 750 Backtest */}
            {activeTab === 'walk-forward' && (
              <WalkForwardMatrix
                comparison={backtestComparison}
                onRunBacktest={handleRunBacktest}
                isRunning={isBacktesting}
              />
            )}

            {/* VIEW 4: Prediction Ledger */}
            {activeTab === 'predictions' && (
              <PredictionLedger
                latestPrediction={latestPrediction}
                history={predictionHistory}
                onExplainWithGemini={handleExplainWithGemini}
                isExplaining={isExplaining}
                onRefreshPrediction={handleRefreshPrediction}
              />
            )}
          </>
        )}
      </main>

      {/* Footer (Anti-Slop: clean copyright and research disclaimer only) */}
      <footer className="mt-auto border-t border-[#1E2638] bg-[#090C10] px-6 py-4 text-xs text-[#8B949E]">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-[#E6EDF3]">NIFTY Sentinel AI</span>
            <span aria-hidden="true">·</span>
            <span>Educational Quantitative Research Terminal</span>
            <span aria-hidden="true">·</span>
            <span>Zero Automated Trading Execution</span>
          </div>
          <div className="font-mono tabular-nums text-[11px]">
            Backend: Python 3.10 + FastAPI (Port 8050) · Gateway: Port 3000
          </div>
        </div>
      </footer>

      {/* System Architecture & Component Manifest Drawer */}
      <SystemAuditDrawer
        isOpen={isAuditOpen}
        onClose={() => setIsAuditOpen(false)}
        status={systemStatus}
        manifest={systemManifest}
      />

      {/* Post-hoc Gemini Explanation Modal */}
      <GeminiExplanationModal
        isOpen={isExplanationOpen}
        onClose={() => setIsExplanationOpen(false)}
        explanation={explanationText}
        isLoading={isExplaining}
        error={explanationError}
      />
    </div>
  );
}
