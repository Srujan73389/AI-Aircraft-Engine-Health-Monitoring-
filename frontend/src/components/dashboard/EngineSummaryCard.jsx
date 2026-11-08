import React from 'react';
import { Plane, Flame, Wind, Gauge, Layers, ShieldCheck, AlertCircle } from 'lucide-react';
import { useEngine } from '../../context/EngineContext';

export const EngineSummaryCard = () => {
  const {
    selectedSubset,
    selectedUnit,
    selectedCycle,
    maxCycles,
    cycleState,
    loading,
    error,
  } = useEngine();

  const state = cycleState?.state_vector || {};

  // Extract core aerothermal parameters strictly from state vector (Zero fabricated fallbacks)
  const t30 = state.T30 ?? state.G8_T30 ?? null;
  const p30 = state.P30 ?? state.G12_P30 ?? null;
  const nf = state.Nf ?? state.G13_Nf ?? null;
  const nc = state.Nc ?? state.G14_Nc ?? null;

  // Additional secondary telemetry
  const t24 = state.T24 ?? state.G7_T24 ?? null;
  const t50 = state.T50 ?? state.G9_T50 ?? null;
  const epr = state.epr ?? state.G15_epr ?? null;
  const bpr = state.BPR ?? state.G20_BPR ?? null;

  // Operating context
  const isMultiRegime = selectedSubset === 'FD002' || selectedSubset === 'FD004';
  const regimeDesc = isMultiRegime ? '6 Altitude/Mach Regimes' : 'Single Sea Level (SLS)';

  return (
    <div className="bg-[#111827] border border-[#1F2937] rounded-xl p-5 shadow-xl shadow-black/20">
      {/* Header Banner: Engine ID, Cycle Status & Provenance */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#1F2937] gap-3">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-blue-950/80 border border-blue-800/60 flex items-center justify-center text-blue-400">
            <Plane className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-white font-mono">
                ENG-{selectedSubset}-U{selectedUnit}
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                ACTIVE
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono">
              Turbofan Gas-Path Digital Twin • {regimeDesc}
            </p>
          </div>
        </div>

        {/* Operational Cycle Indicator Badge */}
        <div className="flex items-center space-x-4 bg-slate-900/90 border border-[#1F2937] px-3.5 py-1.5 rounded-lg">
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Active Cycle</div>
            <div className="text-base font-bold text-cyan-400 font-mono tabular-nums">
              {selectedCycle} <span className="text-xs text-slate-500 font-normal">/ {maxCycles}</span>
            </div>
          </div>
          <div className="h-7 w-px bg-slate-800" />
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Life Consumed</div>
            <div className="text-base font-bold text-slate-200 font-mono tabular-nums">
              {maxCycles > 0 ? ((selectedCycle / maxCycles) * 100).toFixed(1) : '0.0'}%
            </div>
          </div>
        </div>
      </div>

      {/* Loading or Error State Overlays */}
      {loading && (
        <div className="py-2 text-center text-xs text-cyan-400 font-mono flex items-center justify-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span>Synchronizing state vector S({selectedCycle}) from digital twin backend...</span>
        </div>
      )}

      {error && (
        <div className="my-3 p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs font-mono flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Primary Aerothermal Indicators Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5 mt-4">
        {/* Metric 1: HPC Exit Total Temperature (T30) */}
        <div className="bg-slate-900/80 border border-[#1F2937] rounded-lg p-3 hover:border-slate-700 transition-colors">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="flex items-center space-x-1.5 font-mono">
              <Flame className="w-3.5 h-3.5 text-amber-400" />
              <span>T30 (HPC Temp)</span>
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              MEASURED
            </span>
          </div>
          <div className="text-xl font-bold font-mono tabular-nums text-white">
            {typeof t30 === 'number' ? t30.toFixed(2) : (loading ? '...' : '--')}
            <span className="text-xs font-normal text-slate-400 ml-1">°R</span>
          </div>
          <div className="text-[11px] text-slate-400 font-mono mt-0.5">
            Station 3 Combustor Inlet
          </div>
        </div>

        {/* Metric 2: HPC Exit Total Pressure (P30) */}
        <div className="bg-slate-900/80 border border-[#1F2937] rounded-lg p-3 hover:border-slate-700 transition-colors">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="flex items-center space-x-1.5 font-mono">
              <Gauge className="w-3.5 h-3.5 text-cyan-400" />
              <span>P30 (Burner Press)</span>
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              MEASURED
            </span>
          </div>
          <div className="text-xl font-bold font-mono tabular-nums text-white">
            {typeof p30 === 'number' ? p30.toFixed(2) : (loading ? '...' : '--')}
            <span className="text-xs font-normal text-slate-400 ml-1">psia</span>
          </div>
          <div className="text-[11px] text-slate-400 font-mono mt-0.5">
            Combustor Casing Total
          </div>
        </div>

        {/* Metric 3: Physical Fan Speed (Nf) */}
        <div className="bg-slate-900/80 border border-[#1F2937] rounded-lg p-3 hover:border-slate-700 transition-colors">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="flex items-center space-x-1.5 font-mono">
              <Wind className="w-3.5 h-3.5 text-emerald-400" />
              <span>Nf (Fan Speed)</span>
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              MEASURED
            </span>
          </div>
          <div className="text-xl font-bold font-mono tabular-nums text-white">
            {typeof nf === 'number' ? nf.toFixed(1) : (loading ? '...' : '--')}
            <span className="text-xs font-normal text-slate-400 ml-1">rpm</span>
          </div>
          <div className="text-[11px] text-slate-400 font-mono mt-0.5">
            Low-Pressure Spool Speed
          </div>
        </div>

        {/* Metric 4: Physical Core Speed (Nc) */}
        <div className="bg-slate-900/80 border border-[#1F2937] rounded-lg p-3 hover:border-slate-700 transition-colors">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="flex items-center space-x-1.5 font-mono">
              <Gauge className="w-3.5 h-3.5 text-purple-400" />
              <span>Nc (Core Speed)</span>
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              MEASURED
            </span>
          </div>
          <div className="text-xl font-bold font-mono tabular-nums text-white">
            {typeof nc === 'number' ? nc.toFixed(1) : (loading ? '...' : '--')}
            <span className="text-xs font-normal text-slate-400 ml-1">rpm</span>
          </div>
          <div className="text-[11px] text-slate-400 font-mono mt-0.5">
            High-Pressure Core Spool
          </div>
        </div>
      </div>

      {/* Secondary Aerothermal Row: Provenance & Derived Gas-Path Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5 mt-3 pt-3 border-t border-[#1F2937]/70 text-xs font-mono">
        <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/40 rounded border border-[#1F2937]/50">
          <span className="text-slate-400">T24 (LPC Exit):</span>
          <span className="text-slate-200 tabular-nums font-semibold">
            {typeof t24 === 'number' ? `${t24.toFixed(1)} °R` : (loading ? '...' : '--')}
          </span>
        </div>
        <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/40 rounded border border-[#1F2937]/50">
          <span className="text-slate-400">T50 (LPT Exit / EGT):</span>
          <span className="text-slate-200 tabular-nums font-semibold">
            {typeof t50 === 'number' ? `${t50.toFixed(1)} °R` : (loading ? '...' : '--')}
          </span>
        </div>
        <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/40 rounded border border-[#1F2937]/50">
          <span className="text-slate-400">EPR (P50/P2):</span>
          <span className="text-slate-200 tabular-nums font-semibold">
            {typeof epr === 'number' ? epr.toFixed(2) : (loading ? '...' : '--')}
          </span>
        </div>
        <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/40 rounded border border-[#1F2937]/50">
          <span className="text-slate-400">BPR (Bypass Ratio):</span>
          <span className="text-slate-200 tabular-nums font-semibold">
            {typeof bpr === 'number' ? bpr.toFixed(2) : (loading ? '...' : '--')}
          </span>
        </div>
      </div>

      {/* Provenance Badge Legend */}
      <div className="mt-3.5 pt-3 border-t border-[#1F2937] flex flex-wrap items-center justify-between text-[11px] text-slate-400 font-mono gap-2">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
          <span>Variable Provenance Architecture:</span>
        </div>
        <div className="flex items-center space-x-3">
          <span className="flex items-center space-x-1">
            <span className="w-2 h-2 rounded-full bg-blue-500" />
            <span>MEASURED (G6-G26)</span>
          </span>
          <span className="flex items-center space-x-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>DERIVED (G27-G31)</span>
          </span>
          <span className="flex items-center space-x-1">
            <span className="w-2 h-2 rounded-full bg-purple-500" />
            <span>PREDICTED (G32-G37)</span>
          </span>
          <span className="flex items-center space-x-1">
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            <span>SIMULATED (G38-G40)</span>
          </span>
        </div>
      </div>
    </div>
  );
};

export default EngineSummaryCard;
