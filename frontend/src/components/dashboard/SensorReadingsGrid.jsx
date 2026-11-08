import React, { useState } from 'react';
import { Table, Layers, ArrowUpRight, ArrowDownRight, Minus, Filter } from 'lucide-react';
import { useEngine } from '../../context/EngineContext';
import { SENSOR_METADATA, formatSensorValue, getSensorMetadata } from '../../utils/sensorMetadata';

export const SensorReadingsGrid = () => {
  const {
    selectedSubset,
    selectedUnit,
    selectedCycle,
    cycleState,
    sensorTrends,
    loading,
  } = useEngine();

  const [activeTab, setActiveTab] = useState('all');

  const stateVector = cycleState?.state_vector || {};
  const trendsData = sensorTrends?.trends || {};

  // Tabs for categorized filtering
  const tabs = [
    { id: 'all', label: 'All Sensors (21)' },
    { id: 'temperatures', label: 'Gas-Path Temps (°R)' },
    { id: 'pressures', label: 'Pressures (psia)' },
    { id: 'speeds', label: 'Rotor Speeds (rpm)' },
    { id: 'bleed_flow', label: 'Bleed & Flow Ratios' },
    { id: 'performance', label: 'Performance (BPR/farB)' },
  ];

  const filteredSensors = Object.keys(SENSOR_METADATA).filter((sKey) => {
    if (activeTab === 'all') return true;
    return SENSOR_METADATA[sKey].category === activeTab;
  });

  return (
    <div className="bg-[#111827] border border-[#1F2937] rounded-xl p-5 shadow-xl shadow-black/20">
      {/* Header & Category Filter Tabs */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-[#1F2937] gap-3">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-950 border border-emerald-800/80 flex items-center justify-center text-emerald-400">
            <Table className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white font-mono flex items-center space-x-2">
              <span>Full Sensor Telemetry Matrix</span>
              <span className="text-xs font-normal text-slate-400 font-mono">
                (Cycle {selectedCycle} Telemetry with Engineering Units)
              </span>
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Thermodynamic station telemetry and baseline drift relative to initial cycle
            </p>
          </div>
        </div>

        {/* Tab Filter Buttons */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs font-mono">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-2.5 py-1 rounded-md transition-colors border ${
                activeTab === tab.id
                  ? 'bg-cyan-950 border-cyan-700/80 text-cyan-300 font-semibold shadow-sm'
                  : 'bg-slate-900 border-[#1F2937] text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Sensor Table */}
      <div className="overflow-x-auto mt-4">
        <table className="w-full text-left border-collapse text-xs font-mono">
          <thead>
            <tr className="border-b border-[#1F2937] text-slate-400 uppercase text-[11px] bg-slate-900/60">
              <th className="py-2.5 px-3">Channel / ID</th>
              <th className="py-2.5 px-3">Station / Measurement</th>
              <th className="py-2.5 px-3">Category</th>
              <th className="py-2.5 px-3 text-right">Cycle 1 Baseline</th>
              <th className="py-2.5 px-3 text-right">Current (Cycle {selectedCycle})</th>
              <th className="py-2.5 px-3 text-right">Baseline Delta (Δ)</th>
              <th className="py-2.5 px-3 text-center">Provenance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1F2937]/50">
            {filteredSensors.map((sKey) => {
              const meta = SENSOR_METADATA[sKey];

              // Current cycle value: check state vector first, fallback to trends array
              let currVal = stateVector[meta.name] ?? stateVector[sKey];
              if (currVal === undefined && trendsData[sKey]) {
                const idx = Math.max(0, Math.min(selectedCycle - 1, (trendsData[sKey].length || 1) - 1));
                currVal = trendsData[sKey][idx];
              }

              // Baseline cycle 1 value: from trends array index 0
              const baseVal = trendsData[sKey]?.[0] !== undefined
                ? trendsData[sKey][0]
                : currVal;

              const delta = (currVal !== undefined && baseVal !== undefined)
                ? Number(currVal) - Number(baseVal)
                : 0.0;

              const pctDelta = (baseVal && Math.abs(baseVal) > 1e-5)
                ? (delta / Math.abs(baseVal)) * 100
                : 0.0;

              const isSignificant = Math.abs(pctDelta) > 0.05;
              const isPositive = delta > 0;

              return (
                <tr key={sKey} className="hover:bg-slate-900/50 transition-colors">
                  {/* Channel & Code */}
                  <td className="py-2 px-3 font-semibold text-white whitespace-nowrap">
                    <div className="flex items-center space-x-2">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: meta.color }} />
                      <span className="text-cyan-400 font-bold">{meta.name}</span>
                      <span className="text-slate-500 text-[10px]">({sKey})</span>
                    </div>
                  </td>

                  {/* Station & Description */}
                  <td className="py-2 px-3">
                    <div className="text-slate-200">{meta.station}</div>
                    <div className="text-slate-500 text-[10px] truncate max-w-xs">{meta.description}</div>
                  </td>

                  {/* Category */}
                  <td className="py-2 px-3 text-slate-400 capitalize">
                    {meta.category.replace('_', ' ')}
                  </td>

                  {/* Baseline (Cycle 1) */}
                  <td className="py-2 px-3 text-right tabular-nums text-slate-400 whitespace-nowrap">
                    {formatSensorValue(sKey, baseVal)}
                  </td>

                  {/* Current Cycle Value */}
                  <td className="py-2 px-3 text-right tabular-nums font-bold text-white whitespace-nowrap">
                    {formatSensorValue(sKey, currVal)}
                  </td>

                  {/* Baseline Delta (Δ) */}
                  <td className="py-2 px-3 text-right tabular-nums whitespace-nowrap">
                    <div className="flex items-center justify-end space-x-1">
                      {isSignificant ? (
                        isPositive ? (
                          <ArrowUpRight className="w-3.5 h-3.5 text-amber-400" />
                        ) : (
                          <ArrowDownRight className="w-3.5 h-3.5 text-cyan-400" />
                        )
                      ) : (
                        <Minus className="w-3.5 h-3.5 text-slate-600" />
                      )}
                      <span
                        className={
                          !isSignificant
                            ? 'text-slate-500'
                            : isPositive
                            ? 'text-amber-400 font-semibold'
                            : 'text-cyan-400 font-semibold'
                        }
                      >
                        {isPositive ? '+' : ''}
                        {meta.units === 'rpm'
                          ? delta.toFixed(1)
                          : meta.units === '--'
                          ? delta.toFixed(4)
                          : delta.toFixed(2)}{' '}
                        {meta.units !== '--' ? meta.units : ''}
                      </span>
                      <span className="text-[10px] text-slate-500 ml-1">
                        ({isPositive ? '+' : ''}{pctDelta.toFixed(1)}%)
                      </span>
                    </div>
                  </td>

                  {/* Provenance Badge */}
                  <td className="py-2 px-3 text-center whitespace-nowrap">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono">
                      MEASURED
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default SensorReadingsGrid;
