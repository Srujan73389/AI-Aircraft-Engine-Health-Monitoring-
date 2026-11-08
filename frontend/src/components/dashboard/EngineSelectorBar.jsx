import React from 'react';
import { Play, Pause, ChevronLeft, ChevronRight, RotateCcw, Filter, Gauge } from 'lucide-react';
import { useEngine } from '../../context/EngineContext';

export const EngineSelectorBar = () => {
  const {
    selectedSubset,
    selectedUnit,
    selectedCycle,
    maxCycles,
    availableSubsets,
    availableEngines,
    isPlaying,
    loading,
    setSubset,
    setUnit,
    setCycle,
    stepCycle,
    togglePlayReplay,
  } = useEngine();

  const subsetList = availableSubsets.length > 0
    ? availableSubsets.map((s) => s.subset)
    : ['FD001', 'FD002', 'FD003', 'FD004'];

  const engineList = availableEngines.length > 0
    ? availableEngines
    : Array.from({ length: 100 }, (_, i) => i + 1);

  const cycleProgressPct = maxCycles > 0 ? ((selectedCycle / maxCycles) * 100).toFixed(1) : '0.0';

  return (
    <div className="bg-[#111827] border border-[#1F2937] rounded-xl p-4 shadow-lg shadow-black/20">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Top/Left: Subset and Engine Unit Selectors */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Dataset Subset Selector */}
          <div className="flex items-center space-x-2 bg-slate-900/90 border border-[#1F2937] rounded-lg px-3 py-1.5">
            <Filter className="w-4 h-4 text-cyan-400" />
            <label htmlFor="subset-select" className="text-xs uppercase font-mono text-slate-400">
              Subset:
            </label>
            <select
              id="subset-select"
              value={selectedSubset}
              onChange={(e) => setSubset(e.target.value)}
              className="bg-transparent text-sm font-mono font-semibold text-cyan-300 focus:outline-none cursor-pointer"
            >
              {subsetList.map((sub) => (
                <option key={sub} value={sub} className="bg-slate-900 text-white">
                  {sub}
                </option>
              ))}
            </select>
          </div>

          {/* Engine Unit Selector */}
          <div className="flex items-center space-x-2 bg-slate-900/90 border border-[#1F2937] rounded-lg px-3 py-1.5">
            <Gauge className="w-4 h-4 text-cyan-400" />
            <label htmlFor="unit-select" className="text-xs uppercase font-mono text-slate-400">
              Engine Unit:
            </label>
            <select
              id="unit-select"
              value={selectedUnit}
              onChange={(e) => setUnit(e.target.value)}
              className="bg-transparent text-sm font-mono font-semibold text-cyan-300 focus:outline-none cursor-pointer"
            >
              {engineList.map((u) => (
                <option key={u} value={u} className="bg-slate-900 text-white">
                  Unit {u}
                </option>
              ))}
            </select>
          </div>

          {/* Replay Play/Pause and Reset Steppers */}
          <div className="flex items-center space-x-1 bg-slate-900/90 border border-[#1F2937] rounded-lg p-1">
            <button
              onClick={() => stepCycle(-1)}
              disabled={selectedCycle <= 1 || loading}
              title="Step -1 Cycle"
              className="p-1 rounded text-slate-300 hover:text-white hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-transparent"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={togglePlayReplay}
              disabled={loading}
              title={isPlaying ? 'Pause Replay' : 'Play Trajectory Replay'}
              className="p-1 rounded text-cyan-400 hover:text-cyan-300 hover:bg-cyan-950/60 transition-colors"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            <button
              onClick={() => stepCycle(1)}
              disabled={selectedCycle >= maxCycles || loading}
              title="Step +1 Cycle"
              className="p-1 rounded text-slate-300 hover:text-white hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-transparent"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCycle(1)}
              disabled={selectedCycle === 1 || loading}
              title="Reset to Cycle 1"
              className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Right/Bottom: Interactive Cycle Scrubber Slider */}
        <div className="flex-1 max-w-xl flex flex-col space-y-1">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>OPERATING CYCLE:</span>
            <div className="flex items-center space-x-2">
              <span className="text-cyan-400 font-bold tabular-nums text-sm">
                Cycle {selectedCycle}
              </span>
              <span className="text-slate-500">/</span>
              <span className="tabular-nums text-slate-300">
                {maxCycles} Max ({cycleProgressPct}%)
              </span>
            </div>
          </div>

          <div className="relative flex items-center">
            <input
              type="range"
              min="1"
              max={maxCycles}
              value={selectedCycle}
              onChange={(e) => setCycle(Number(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400 focus:outline-none"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default EngineSelectorBar;
