import React, { useState, useMemo } from 'react';
import { OHLCVRecord } from '../types';

interface PriceChartProps {
  data: OHLCVRecord[];
}

export const PriceChart: React.FC<PriceChartProps> = ({ data }) => {
  const [range, setRange] = useState<'6M' | '1Y' | 'ALL'>('1Y');
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    if (range === '6M') return data.slice(-126);
    if (range === '1Y') return data.slice(-252);
    return data;
  }, [data, range]);

  // Compute SMA 20 and SMA 50
  const chartData = useMemo(() => {
    return filteredData.map((d, i, arr) => {
      let sma20: number | null = null;
      let sma50: number | null = null;
      if (i >= 19) {
        const slice20 = arr.slice(i - 19, i + 1);
        sma20 = slice20.reduce((acc, curr) => acc + curr.close, 0) / 20;
      }
      if (i >= 49) {
        const slice50 = arr.slice(i - 49, i + 1);
        sma50 = slice50.reduce((acc, curr) => acc + curr.close, 0) / 50;
      }
      return { ...d, sma20, sma50 };
    });
  }, [filteredData]);

  if (!chartData || chartData.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center bg-[#121721] border border-[#1E2638] rounded-lg text-xs text-[#8B949E]">
        No historical price records loaded. Click 'Sync Market Data' to ingest.
      </div>
    );
  }

  // Min and max for scaling
  const prices = chartData.map((d) => d.close);
  const minPrice = Math.min(...chartData.map((d) => d.low)) * 0.99;
  const maxPrice = Math.max(...chartData.map((d) => d.high)) * 1.01;
  const maxVol = Math.max(...chartData.map((d) => d.volume || 1));

  const width = 1000;
  const height = 320;
  const padding = { top: 20, right: 70, bottom: 40, left: 10 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const volHeight = 50;

  const getX = (idx: number) => padding.left + (idx / Math.max(1, chartData.length - 1)) * chartWidth;
  const getY = (val: number) => padding.top + chartHeight - ((val - minPrice) / (maxPrice - minPrice)) * chartHeight;
  const getVolY = (vol: number) => height - padding.bottom - (vol / maxVol) * volHeight;

  // Price path
  const pricePoints = chartData.map((d, i) => `${getX(i)},${getY(d.close)}`).join(' ');
  const areaPoints = `${getX(0)},${padding.top + chartHeight} ${pricePoints} ${getX(chartData.length - 1)},${padding.top + chartHeight}`;

  // SMA paths
  const sma20Points = chartData
    .filter((d) => d.sma20 !== null)
    .map((d) => `${getX(chartData.indexOf(d))},${getY(d.sma20!)}`)
    .join(' ');

  const sma50Points = chartData
    .filter((d) => d.sma50 !== null)
    .map((d) => `${getX(chartData.indexOf(d))},${getY(d.sma50!)}`)
    .join(' ');

  // Price Y-axis labels
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((pct) => minPrice + pct * (maxPrice - minPrice));

  const hoveredItem = hoverIdx !== null && chartData[hoverIdx] ? chartData[hoverIdx] : chartData[chartData.length - 1];

  return (
    <div className="p-5 bg-[#121721] border border-[#1E2638] rounded-lg">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-3">
        <div>
          <div className="flex items-center gap-3">
            <span className="text-lg font-bold font-mono tabular-nums text-[#E6EDF3]">
              ₹{hoveredItem.close.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
            </span>
            <span
              className={`text-xs font-mono tabular-nums ${
                hoveredItem.close >= hoveredItem.open ? 'text-[#3FB950]' : 'text-[#F85149]'
              }`}
            >
              {hoveredItem.close >= hoveredItem.open ? '+' : ''}
              {(((hoveredItem.close - hoveredItem.open) / hoveredItem.open) * 100).toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-[#8B949E] font-mono tabular-nums mt-0.5">
            <span>Date: {hoveredItem.date}</span>
            <span aria-hidden="true">·</span>
            <span>O: {hoveredItem.open.toFixed(1)}</span>
            <span aria-hidden="true">·</span>
            <span>H: {hoveredItem.high.toFixed(1)}</span>
            <span aria-hidden="true">·</span>
            <span>L: {hoveredItem.low.toFixed(1)}</span>
            {hoveredItem.sma20 && (
              <>
                <span aria-hidden="true">·</span>
                <span className="text-[#58A6FF]">SMA20: {hoveredItem.sma20.toFixed(1)}</span>
              </>
            )}
            {hoveredItem.sma50 && (
              <>
                <span aria-hidden="true">·</span>
                <span className="text-[#D29922]">SMA50: {hoveredItem.sma50.toFixed(1)}</span>
              </>
            )}
          </div>
        </div>

        {/* Segmented Range Filter Tabs */}
        <div className="flex items-center gap-1 p-0.5 bg-[#090C10] border border-[#1E2638] rounded">
          {(['6M', '1Y', 'ALL'] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`px-2.5 py-1 text-xs font-mono transition-colors rounded ${
                range === r ? 'bg-[#1E2638] text-[#E6EDF3] font-medium' : 'text-[#8B949E] hover:text-[#E6EDF3]'
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      <div className="relative w-full overflow-hidden">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-64 select-none"
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = (e.clientX - rect.left) / rect.width;
            const idx = Math.min(
              chartData.length - 1,
              Math.max(0, Math.round(((relX * width - padding.left) / chartWidth) * (chartData.length - 1)))
            );
            setHoverIdx(idx);
          }}
          onMouseLeave={() => setHoverIdx(null)}
        >
          {/* Horizontal Grid Lines & Y-ticks */}
          {yTicks.map((val, idx) => {
            const y = getY(val);
            return (
              <g key={idx}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#1E2638" strokeDasharray="3 3" />
                <text
                  x={width - padding.right + 8}
                  y={y + 3}
                  fill="#8B949E"
                  fontSize="10"
                  fontFamily="monospace"
                  textAnchor="start"
                >
                  {val.toFixed(0)}
                </text>
              </g>
            );
          })}

          {/* Volume bars at bottom */}
          {chartData.map((d, i) => {
            const x = getX(i);
            const y = getVolY(d.volume);
            const isUp = d.close >= d.open;
            return (
              <line
                key={`vol-${i}`}
                x1={x}
                y1={height - padding.bottom}
                x2={x}
                y2={y}
                stroke={isUp ? 'rgba(63, 185, 80, 0.25)' : 'rgba(248, 81, 73, 0.25)'}
                strokeWidth={Math.max(1, chartWidth / chartData.length - 1)}
              />
            );
          })}

          {/* Shaded Price Area */}
          <polygon points={areaPoints} fill="rgba(88, 166, 255, 0.04)" />

          {/* Close Price Line */}
          <polyline fill="none" stroke="#58A6FF" strokeWidth="1.5" points={pricePoints} />

          {/* SMA 20 (Cyan) */}
          {sma20Points && <polyline fill="none" stroke="#79c0ff" strokeWidth="1.2" strokeDasharray="4 2" points={sma20Points} />}

          {/* SMA 50 (Gold) */}
          {sma50Points && <polyline fill="none" stroke="#d29922" strokeWidth="1.2" points={sma50Points} />}

          {/* Interactive Crosshair */}
          {hoverIdx !== null && (
            <g>
              <line
                x1={getX(hoverIdx)}
                y1={padding.top}
                x2={getX(hoverIdx)}
                y2={height - padding.bottom}
                stroke="#8B949E"
                strokeWidth="1"
                strokeDasharray="2 2"
              />
              <circle cx={getX(hoverIdx)} cy={getY(chartData[hoverIdx].close)} r="4" fill="#58A6FF" stroke="#090C10" strokeWidth="2" />
            </g>
          )}

          {/* Date Axis X Ticks */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
            const i = Math.floor(pct * (chartData.length - 1));
            return (
              <text
                key={idx}
                x={getX(i)}
                y={height - 12}
                fill="#8B949E"
                fontSize="10"
                fontFamily="monospace"
                textAnchor="middle"
              >
                {chartData[i].date.slice(5)}
              </text>
            );
          })}
        </svg>
      </div>

      <div className="flex items-center justify-between mt-2 pt-2 border-t border-[#1E2638] text-[11px] text-[#8B949E] font-mono">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 bg-[#58A6FF] inline-block" /> Close
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 bg-[#79c0ff] border-t border-dashed border-[#79c0ff] inline-block" /> SMA 20
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 bg-[#d29922] inline-block" /> SMA 50
          </span>
        </div>
        <div>NIFTY 50 (NSE India) · Daily OHLC</div>
      </div>
    </div>
  );
};
