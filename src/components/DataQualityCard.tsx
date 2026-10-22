import React from 'react';
import { ShieldCheck, AlertTriangle, Database, Calendar } from 'lucide-react';
import { DataQualityAudit } from '../types';

interface DataQualityCardProps {
  audit: DataQualityAudit | null;
  totalRecords: number;
  latestSourceTs?: string;
  dataSource?: string;
}

export const DataQualityCard: React.FC<DataQualityCardProps> = ({
  audit,
  totalRecords,
  latestSourceTs,
  dataSource = 'yfinance',
}) => {
  if (!audit) {
    return (
      <div className="p-4 bg-[#121721] border border-[#1E2638] rounded-lg">
        <div className="flex items-center gap-2 text-xs text-[#8B949E]">
          <Database className="w-4 h-4 text-[#58A6FF]" />
          <span>Repository Status: Ready for synchronization</span>
        </div>
      </div>
    );
  }

  const isNominal = audit.anomaly_count === 0 && audit.status === 'VALID';

  return (
    <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-[#1E2638]">
        <div>
          <div className="flex items-center gap-2">
            {isNominal ? (
              <ShieldCheck className="w-4 h-4 text-[#3FB950]" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-[#D29922]" />
            )}
            <h3 className="text-sm font-semibold text-[#E6EDF3]">Data Ingestion & Integrity Audit</h3>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-[#8B949E] font-mono tabular-nums">
            <span>Provider: {dataSource}</span>
            <span aria-hidden="true">·</span>
            <span>Table: nifty_daily_ohlcv</span>
            <span aria-hidden="true">·</span>
            <span>Range: {audit.start_date} to {audit.end_date}</span>
            <span aria-hidden="true">·</span>
            <span>Sessions: {audit.total_sessions}</span>
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs text-[#8B949E]">Audit Integrity Score</div>
          <div className="text-lg font-bold font-mono tabular-nums text-[#3FB950]">
            {isNominal ? '100% NOMINAL' : `${audit.anomaly_count} ANOMALIES`}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4">
        <div>
          <div className="text-[11px] text-[#8B949E]">Historical Span</div>
          <div className="text-sm font-mono tabular-nums text-[#E6EDF3] mt-0.5">
            {audit.total_sessions} Days (~5 Years)
          </div>
        </div>
        <div>
          <div className="text-[11px] text-[#8B949E]">NSE Holidays Identified</div>
          <div className="text-sm font-mono tabular-nums text-[#E6EDF3] mt-0.5">
            {audit.holiday_matches} Sessions
          </div>
        </div>
        <div>
          <div className="text-[11px] text-[#8B949E]">Non-Trading Weekdays</div>
          <div className="text-sm font-mono tabular-nums text-[#E6EDF3] mt-0.5">
            {audit.missing_weekday_count} Filtered
          </div>
        </div>
        <div>
          <div className="text-[11px] text-[#8B949E]">Source Timestamp</div>
          <div className="text-xs font-mono tabular-nums text-[#8B949E] mt-0.5 truncate" title={audit.audit_ts}>
            {audit.audit_ts.slice(0, 19).replace('T', ' ')} UTC
          </div>
        </div>
      </div>

      {audit.missing_weekday_count > 0 && audit.details?.missing_weekdays_sample && (
        <div className="mt-3 pt-3 border-t border-[#1E2638]/60 text-[11px] text-[#8B949E]">
          <span className="text-[#E6EDF3] font-medium">Calendar Diagnostic:</span> Missing weekday dates cross-referenced against NSE trading calendar. Sample holiday sessions: {audit.details.missing_weekdays_sample.slice(0, 5).join(', ')}.
        </div>
      )}
    </div>
  );
};
