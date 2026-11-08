import React from 'react';
import {
  Timer,
  TrendingDown,
  Cpu,
  Layers,
  ShieldCheck,
  AlertTriangle,
  Info,
  CheckCircle2,
  HelpCircle,
} from 'lucide-react';
import { useEngine } from '../../context/EngineContext';

export const PrognosticsRULPanel = () => {
  const {
    selectedSubset,
    selectedUnit,
    selectedCycle,
    maxCycles,
    prognosticsData,
    prognosticsLoading,
  } = useEngine();

  // Extract prognostic indicators strictly from real model inference (Zero fabricated fallbacks)
  const consensusRul = prognosticsData?.consensus_rul ?? null;
  const xgbRul = prognosticsData?.predicted_rul_xgboost ?? null;
  const lstmRul = prognosticsData?.predicted_rul_lstm ?? null;
  const spread = prognosticsData?.prognostic_spread ?? (
    (xgbRul !== null && lstmRul !== null) ? Math.abs(xgbRul - lstmRul) : null
  );
  const confidenceScore = prognosticsData?.confidence_score ?? null;

  // Confidence Rating Category
  let confidenceLabel = prognosticsLoading ? 'COMPUTING...' : 'AWAITING INFERENCE';
  let confidenceBadgeClass = 'bg-slate-800 text-slate-400 border-slate-700';
  if (confidenceScore !== null) {
    if (confidenceScore < 0.60) {
      confidenceLabel = 'LOW';
      confidenceBadgeClass = 'bg-rose-500/10 text-rose-400 border-rose-500/20';
    } else if (confidenceScore < 0.80) {
      confidenceLabel = 'MODERATE';
      confidenceBadgeClass = 'bg-amber-500/10 text-amber-400 border-amber-500/20';
    } else {
      confidenceLabel = 'HIGH';
      confidenceBadgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
    }
  }

  // Feature Tier mapping based on dataset subset
  const featureTier = selectedSubset === 'FD001'
    ? 'Tier 1 Core-14 Telemetry'
    : 'Tier 2 Multi-Regime 24 Features';

  // SVG Trajectory Curve calculations (guarded against uncomputed inference)
  const totalLifespan = consensusRul !== null ? (selectedCycle + consensusRul) : (maxCycles || 200);
  const svgWidth = 600;
  const svgHeight = 160;
  const pad = { top: 20, right: 35, bottom: 30, left: 45 };
  const pWidth = svgWidth - pad.left - pad.right;
  const pHeight = svgHeight - pad.top - pad.bottom;

  // Trajectory points: from cycle 1 down to EOL
  const currX = pad.left + (selectedCycle / Math.max(1, totalLifespan)) * pWidth;
  const currY = pad.top + (1 - ((consensusRul ?? 0) / Math.max(1, totalLifespan))) * pHeight;

  const pathStart = `M ${pad.left},${pad.top}`;
  const pathCurrent = `L ${currX},${currY}`;
  const pathEnd = `L ${pad.left + pWidth},${pad.top + pHeight}`;

  return (
    <div className="bg-[#111827] border border-[#1F2937] rounded-xl p-5 shadow-xl shadow-black/20 space-y-5">
      {/* ========================================================================= */}
      {/* SECTION 1: Primary Prognostic Overview Banner */}
      {/* ========================================================================= */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between pb-4 border-b border-[#1F2937] gap-4">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-950 border border-cyan-800/80 flex items-center justify-center text-cyan-400">
            <Timer className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-base font-bold text-white font-mono">
                Consensus Remaining Useful Life (RUL)
              </h3>
              <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                Dual-Model Ensemble
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono">
              Statistical operational lifespan forecast evaluated at flight cycle {selectedCycle}
            </p>
          </div>
        </div>

        {/* Primary RUL Numerical Display & Confidence Gauge */}
        <div className="flex flex-wrap items-center gap-4 bg-slate-900/90 border border-[#1F2937] p-3 rounded-lg">
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Consensus Prediction</div>
            <div className="flex items-baseline space-x-1.5">
              <span className="text-2xl font-bold font-mono text-white tabular-nums">
                {typeof consensusRul === 'number'
                  ? consensusRul.toFixed(1)
                  : (prognosticsLoading ? 'COMPUTING...' : '--')}
              </span>
              <span className="text-xs font-mono text-cyan-400 font-semibold uppercase">
                Operating Cycles
              </span>
            </div>
          </div>

          <div className="h-8 w-px bg-slate-800 hidden sm:block" />

          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Model Spread (|Δ|)</div>
            <div className="flex items-baseline space-x-1">
              <span className="text-base font-bold font-mono text-slate-200 tabular-nums">
                {typeof spread === 'number'
                  ? `±${spread.toFixed(1)}`
                  : (prognosticsLoading ? '...' : '--')}
              </span>
              <span className="text-[11px] font-mono text-slate-400">Operating Cycles</span>
            </div>
          </div>

          <div className="h-8 w-px bg-slate-800 hidden sm:block" />

          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Confidence Rating</div>
            <div className={`mt-0.5 px-2 py-0.5 rounded text-xs font-mono font-bold border inline-flex items-center space-x-1 ${confidenceBadgeClass}`}>
              <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
              <span>
                {confidenceLabel}
                {typeof confidenceScore === 'number' ? ` (${(confidenceScore * 100).toFixed(0)}%)` : ''}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Loading Indicator */}
      {prognosticsLoading && (
        <div className="py-2 text-center text-xs text-cyan-400 font-mono flex items-center justify-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span>Executing champion XGBoost and Bi-LSTM inference for cycle {selectedCycle}...</span>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SECTION 2: Dual Model Breakdown Cards */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Card A: XGBoost Gradient Boosted Trees */}
        <div className="bg-slate-900/80 border border-[#1F2937] rounded-lg p-4">
          <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]/70">
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span className="font-mono font-bold text-sm text-white">
                XGBoost Regressor (Champion Trees)
              </span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              {featureTier}
            </span>
          </div>

          <div className="flex items-baseline justify-between mt-3">
            <div>
              <div className="text-[10px] font-mono uppercase text-slate-400">Model Estimated RUL</div>
              <div className="flex items-baseline space-x-1.5 mt-0.5">
                <span className="text-xl font-bold font-mono text-white tabular-nums">
                  {typeof xgbRul === 'number'
                    ? xgbRul.toFixed(1)
                    : (prognosticsLoading ? 'COMPUTING...' : '--')}
                </span>
                <span className="text-xs font-mono text-cyan-300">Operating Cycles</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] font-mono uppercase text-slate-400">Algorithm Class</div>
              <div className="text-xs font-mono text-slate-300 mt-0.5">Gradient Boosted Ensembles</div>
            </div>
          </div>

          <p className="mt-2.5 pt-2 border-t border-[#1F2937]/50 text-[11px] font-mono text-slate-400">
            Direct tabular degradation regression based on 14-sensor instantaneous feature vector.
          </p>
        </div>

        {/* Card B: PyTorch Bi-LSTM Recurrent Neural Network */}
        <div className="bg-slate-900/80 border border-[#1F2937] rounded-lg p-4">
          <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]/70">
            <div className="flex items-center space-x-2">
              <Layers className="w-4 h-4 text-purple-400" />
              <span className="font-mono font-bold text-sm text-white">
                PyTorch Bi-LSTM (Recurrent Deep Net)
              </span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
              30-Cycle Sequence Window
            </span>
          </div>

          <div className="flex items-baseline justify-between mt-3">
            <div>
              <div className="text-[10px] font-mono uppercase text-slate-400">Model Estimated RUL</div>
              <div className="flex items-baseline space-x-1.5 mt-0.5">
                <span className="text-xl font-bold font-mono text-white tabular-nums">
                  {typeof lstmRul === 'number'
                    ? lstmRul.toFixed(1)
                    : (prognosticsLoading ? 'COMPUTING...' : '--')}
                </span>
                <span className="text-xs font-mono text-purple-300">Operating Cycles</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] font-mono uppercase text-slate-400">Temporal Architecture</div>
              <div className="text-xs font-mono text-slate-300 mt-0.5">Bidirectional Recurrent LSTM</div>
            </div>
          </div>

          <p className="mt-2.5 pt-2 border-t border-[#1F2937]/50 text-[11px] font-mono text-slate-400">
            Captures cumulative temporal memory and degradation rates over sliding temporal window.
          </p>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 3: RUL Trajectory Trend Curve */}
      {/* ========================================================================= */}
      <div className="bg-slate-900/60 border border-[#1F2937] rounded-lg p-4">
        <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]/70 mb-2">
          <div className="flex items-center space-x-2">
            <TrendingDown className="w-4 h-4 text-cyan-400" />
            <h4 className="font-mono font-bold text-sm text-white">
              Remaining Useful Life Degradation Trajectory
            </h4>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Projected Total Life: ~{consensusRul !== null ? Math.round(totalLifespan) : '--'} Operating Cycles
          </span>
        </div>

        {consensusRul === null ? (
          <div className="h-40 rounded bg-slate-950/80 border border-[#1F2937] flex flex-col items-center justify-center text-slate-500 font-mono text-xs">
            <Timer className="w-6 h-6 mb-2 animate-pulse text-slate-600" />
            <span>
              {prognosticsLoading
                ? 'Executing champion XGBoost and Bi-LSTM models to compute trajectory...'
                : 'Awaiting model inference telemetry'}
            </span>
          </div>
        ) : (
          <div className="relative bg-slate-950/80 rounded border border-[#1F2937] overflow-hidden">
            <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto">
              {/* Background grid */}
              <g className="stroke-[#1F2937]" strokeDasharray="3 3">
                <line x1={pad.left} y1={pad.top} x2={svgWidth - pad.right} y2={pad.top} strokeWidth="1" />
                <line x1={pad.left} y1={pad.top + pHeight * 0.5} x2={svgWidth - pad.right} y2={pad.top + pHeight * 0.5} strokeWidth="1" />
                <line x1={pad.left} y1={svgHeight - pad.bottom} x2={svgWidth - pad.right} y2={svgHeight - pad.bottom} strokeWidth="1" />
              </g>

              {/* Past Elapsed Trajectory Line (Solid Cyan) */}
              <path
                d={`${pathStart} ${pathCurrent}`}
                fill="none"
                stroke="#06B6D4"
                strokeWidth="2.5"
              />

              {/* Future Projected Remaining Life Line (Dashed Purple) */}
              <path
                d={`M ${currX},${currY} ${pathEnd}`}
                fill="none"
                stroke="#A855F7"
                strokeWidth="2"
                strokeDasharray="4 4"
              />

              {/* Current Cycle Marker */}
              <circle cx={currX} cy={currY} r="5" fill="#06B6D4" stroke="#FFFFFF" strokeWidth="2" />

              {/* Cursor Label Pill */}
              <g>
                <rect
                  x={Math.max(pad.left, currX - 45)}
                  y={Math.max(pad.top, currY - 26)}
                  width="90"
                  height="18"
                  rx="4"
                  fill="#111827"
                  stroke="#06B6D4"
                  strokeWidth="1"
                />
                <text
                  x={Math.max(pad.left + 45, currX)}
                  y={Math.max(pad.top + 13, currY - 13)}
                  fill="#06B6D4"
                  fontSize="9"
                  fontWeight="bold"
                  fontFamily="JetBrains Mono, monospace"
                  textAnchor="middle"
                >
                  C{selectedCycle}: {consensusRul.toFixed(0)} Cycles
                </text>
              </g>

              {/* Axis Labels */}
              <g fill="#64748B" fontSize="10" fontFamily="JetBrains Mono, monospace">
                <text x={pad.left} y={svgHeight - 10} textAnchor="start">Cycle 1 (New)</text>
                <text x={svgWidth - pad.right} y={svgHeight - 10} textAnchor="end">EOL (0 Cycles RUL)</text>
                <text x={pad.left - 8} y={pad.top + 4} textAnchor="end">Max</text>
                <text x={pad.left - 8} y={svgHeight - pad.bottom} textAnchor="end">0</text>
              </g>
            </svg>
          </div>
        )}
      </div>

      {/* ========================================================================= */}
      {/* SECTION 4: Mandatory Academic Estimate Disclaimer Box */}
      {/* ========================================================================= */}
      <div className="p-3.5 rounded-lg bg-cyan-950/30 border border-cyan-800/60 text-cyan-300 text-xs font-mono flex items-start space-x-2.5">
        <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-bold uppercase tracking-wider text-cyan-200">
            Prognostic Estimate Notice • Statistical Lifetime Projection
          </div>
          <p className="text-cyan-300/90 leading-relaxed text-[11px]">
            PROGNOSTIC ESTIMATE NOTICE: Remaining Useful Life (RUL) values displayed are statistical machine learning estimates measured in discrete flight cycles. They do not constitute an airworthiness dispatch authorization or a certified mechanical life limit.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PrognosticsRULPanel;
