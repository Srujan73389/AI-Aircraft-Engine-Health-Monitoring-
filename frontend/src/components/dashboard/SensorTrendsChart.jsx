import React, { useState, useMemo, useRef } from 'react';
import { LineChart, Activity, Crosshair, Eye, EyeOff } from 'lucide-react';
import { useEngine } from '../../context/EngineContext';
import { SENSOR_METADATA, formatSensorValue, getSensorMetadata } from '../../utils/sensorMetadata';

export const SensorTrendsChart = () => {
  const {
    selectedSubset,
    selectedUnit,
    selectedCycle,
    maxCycles,
    sensorTrends,
    trendsLoading,
    setCycle,
  } = useEngine();

  // Default active sensors for visualization
  const [activeSensors, setActiveSensors] = useState(['sensor_3', 'sensor_7', 'sensor_4', 'sensor_9']); // T30, P30, T50, Nc
  const [hoveredCycle, setHoveredCycle] = useState(null);
  const svgRef = useRef(null);

  const toggleSensor = (sensorKey) => {
    setActiveSensors((prev) => {
      if (prev.includes(sensorKey)) {
        if (prev.length === 1) return prev; // Keep at least 1 sensor visible
        return prev.filter((k) => k !== sensorKey);
      } else {
        return [...prev, sensorKey];
      }
    });
  };

  const selectPreset = (keys) => {
    setActiveSensors(keys);
  };

  // Dimensions for SVG viewport
  const width = 800;
  const height = 300;
  const padding = { top: 30, right: 35, bottom: 40, left: 60 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const totalCycles = sensorTrends?.total_cycles || maxCycles || 1;
  const cycles = sensorTrends?.cycle_history || Array.from({ length: totalCycles }, (_, i) => i + 1);
  const trendsData = sensorTrends?.trends || {};

  // Compute normalized series for overlapping visualization
  const seriesInfo = useMemo(() => {
    return activeSensors.map((sensorKey) => {
      const meta = getSensorMetadata(sensorKey);
      const rawValues = trendsData[sensorKey] || [];
      const validVals = rawValues.filter((v) => v !== null && !isNaN(v));
      const minVal = validVals.length > 0 ? Math.min(...validVals) : 0;
      const maxVal = validVals.length > 0 ? Math.max(...validVals) : 1;
      const span = maxVal - minVal || 1;

      const points = rawValues.map((v, i) => {
        const c = cycles[i] || (i + 1);
        const x = padding.left + ((c - 1) / Math.max(1, totalCycles - 1)) * plotWidth;
        const normY = (v - minVal) / span;
        const y = padding.top + (1 - normY) * plotHeight;
        return { x, y, cycle: c, value: v };
      });

      const pathString = points.length > 0
        ? points.reduce((acc, pt, i) => (i === 0 ? `M ${pt.x},${pt.y}` : `${acc} L ${pt.x},${pt.y}`), '')
        : '';

      return {
        key: sensorKey,
        meta,
        minVal,
        maxVal,
        points,
        pathString,
        color: meta?.color || '#06B6D4',
      };
    });
  }, [activeSensors, trendsData, cycles, totalCycles, plotWidth, plotHeight, padding.left, padding.top]);

  // Cursor coordinates for selectedCycle
  const cursorX = padding.left + ((selectedCycle - 1) / Math.max(1, totalCycles - 1)) * plotWidth;

  // Hovered cycle resolution
  const handleMouseMove = (e) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const boundedX = Math.max(padding.left, Math.min(clientX, width - padding.right));
    const fraction = (boundedX - padding.left) / plotWidth;
    const approxCycle = Math.round(1 + fraction * (totalCycles - 1));
    setHoveredCycle(Math.max(1, Math.min(approxCycle, totalCycles)));
  };

  const handleClick = (e) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const boundedX = Math.max(padding.left, Math.min(clientX, width - padding.right));
    const fraction = (boundedX - padding.left) / plotWidth;
    const clickedCycle = Math.round(1 + fraction * (totalCycles - 1));
    setCycle(Math.max(1, Math.min(clickedCycle, totalCycles)));
  };

  const activeInspectCycle = hoveredCycle || selectedCycle;
  const inspectCycleIdx = Math.max(0, Math.min(activeInspectCycle - 1, totalCycles - 1));

  return (
    <div className="bg-[#111827] border border-[#1F2937] rounded-xl p-5 shadow-xl shadow-black/20">
      {/* Chart Top Header & Presets */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-[#1F2937] gap-3">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-800/80 flex items-center justify-center text-cyan-400">
            <LineChart className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white font-mono flex items-center space-x-2">
              <span>Multi-Sensor Trajectory Trends</span>
              <span className="text-xs font-normal text-slate-400 font-mono">
                (Full Engine Run-to-Failure Lifespan)
              </span>
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Hover to inspect values • Click to reposition operational cycle scrubber
            </p>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="flex items-center space-x-1.5 text-xs font-mono">
          <span className="text-slate-400 mr-1 hidden sm:inline">Presets:</span>
          <button
            onClick={() => selectPreset(['sensor_3', 'sensor_7', 'sensor_4', 'sensor_9'])}
            className="px-2 py-1 rounded bg-slate-900 border border-[#1F2937] text-slate-300 hover:text-cyan-400 hover:border-cyan-500/40 text-[11px]"
          >
            Core (T30, P30, T50, Nc)
          </button>
          <button
            onClick={() => selectPreset(['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4'])}
            className="px-2 py-1 rounded bg-slate-900 border border-[#1F2937] text-slate-300 hover:text-cyan-400 hover:border-cyan-500/40 text-[11px]"
          >
            Temperatures
          </button>
          <button
            onClick={() => selectPreset(['sensor_5', 'sensor_6', 'sensor_7', 'sensor_11'])}
            className="px-2 py-1 rounded bg-slate-900 border border-[#1F2937] text-slate-300 hover:text-cyan-400 hover:border-cyan-500/40 text-[11px]"
          >
            Pressures
          </button>
          <button
            onClick={() => selectPreset(['sensor_8', 'sensor_9', 'sensor_13', 'sensor_14'])}
            className="px-2 py-1 rounded bg-slate-900 border border-[#1F2937] text-slate-300 hover:text-cyan-400 hover:border-cyan-500/40 text-[11px]"
          >
            Speeds
          </button>
        </div>
      </div>

      {/* Sensor Toggle Pills */}
      <div className="flex flex-wrap gap-2 py-3 border-b border-[#1F2937]/70">
        {Object.keys(SENSOR_METADATA).map((sKey) => {
          const meta = SENSOR_METADATA[sKey];
          const isSelected = activeSensors.includes(sKey);
          return (
            <button
              key={sKey}
              onClick={() => toggleSensor(sKey)}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs font-mono transition-all border ${
                isSelected
                  ? 'bg-slate-900 text-white shadow-sm'
                  : 'bg-slate-950/60 text-slate-400 border-transparent hover:border-slate-800'
              }`}
              style={{ borderColor: isSelected ? meta.color : undefined }}
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: isSelected ? meta.color : '#4B5563' }}
              />
              <span className="font-semibold">{meta.name}</span>
              <span className="text-[10px] text-slate-400">({meta.units})</span>
            </button>
          );
        })}
      </div>

      {/* Main SVG Chart Canvas */}
      <div className="relative mt-4 bg-slate-950/80 rounded-lg border border-[#1F2937] overflow-hidden select-none">
        {trendsLoading && (
          <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm z-20 flex items-center justify-center space-x-2 text-cyan-400 font-mono text-xs">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping" />
            <span>Loading 21-channel trajectory telemetry from backend...</span>
          </div>
        )}

        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredCycle(null)}
          onClick={handleClick}
        >
          {/* Grid Background Lines */}
          <g className="stroke-[#1F2937]" strokeDasharray="3 3">
            {[0, 0.25, 0.5, 0.75, 1.0].map((frac, i) => {
              const y = padding.top + frac * plotHeight;
              return <line key={`h-${i}`} x1={padding.left} y1={y} x2={width - padding.right} y2={y} strokeWidth="1" />;
            })}
            {[0, 0.25, 0.5, 0.75, 1.0].map((frac, i) => {
              const x = padding.left + frac * plotWidth;
              const c = Math.round(1 + frac * (totalCycles - 1));
              return (
                <g key={`v-${i}`}>
                  <line x1={x} y1={padding.top} x2={x} y2={height - padding.bottom} strokeWidth="1" />
                  <text
                    x={x}
                    y={height - padding.bottom + 18}
                    fill="#64748B"
                    fontSize="11"
                    fontFamily="JetBrains Mono, monospace"
                    textAnchor="middle"
                  >
                    C{c}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Left Y-Axis Normalized Scale Labels */}
          <g fill="#64748B" fontSize="10" fontFamily="JetBrains Mono, monospace" textAnchor="end">
            <text x={padding.left - 8} y={padding.top + 4}>100%</text>
            <text x={padding.left - 8} y={padding.top + plotHeight * 0.5 + 4}>50%</text>
            <text x={padding.left - 8} y={height - padding.bottom}>0%</text>
          </g>

          {/* Sensor Trajectory Curves */}
          {seriesInfo.map((s) => (
            <path
              key={s.key}
              d={s.pathString}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.9"
            />
          ))}

          {/* Active Operating Cycle Reference Cursor Line (selectedCycle) */}
          <g>
            <line
              x1={cursorX}
              y1={padding.top}
              x2={cursorX}
              y2={height - padding.bottom}
              stroke="#06B6D4"
              strokeWidth="2"
              strokeDasharray="4 4"
            />
            {/* Top Cursor Tag */}
            <rect
              x={cursorX - 26}
              y={padding.top - 20}
              width="52"
              height="18"
              rx="4"
              fill="#06B6D4"
            />
            <text
              x={cursorX}
              y={padding.top - 7}
              fill="#0B0F19"
              fontSize="10"
              fontWeight="bold"
              fontFamily="JetBrains Mono, monospace"
              textAnchor="middle"
            >
              C{selectedCycle}
            </text>
          </g>

          {/* Hover Crosshair Line */}
          {hoveredCycle && hoveredCycle !== selectedCycle && (
            <g>
              {(() => {
                const hoverX = padding.left + ((hoveredCycle - 1) / Math.max(1, totalCycles - 1)) * plotWidth;
                return (
                  <>
                    <line
                      x1={hoverX}
                      y1={padding.top}
                      x2={hoverX}
                      y2={height - padding.bottom}
                      stroke="#94A3B8"
                      strokeWidth="1"
                      strokeDasharray="2 2"
                    />
                    <circle cx={hoverX} cy={padding.top} r="3" fill="#94A3B8" />
                  </>
                );
              })()}
            </g>
          )}
        </svg>

        {/* Dynamic Telemetry Tooltip Inspector Bar */}
        <div className="bg-slate-900 border-t border-[#1F2937] px-4 py-2 flex flex-wrap items-center justify-between text-xs font-mono gap-2">
          <div className="flex items-center space-x-2">
            <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Inspecting:</span>
            <span className="text-cyan-400 font-bold tabular-nums">
              Flight Cycle {activeInspectCycle} of {totalCycles}
            </span>
            {hoveredCycle && (
              <span className="text-[10px] text-slate-500">(Click to seek)</span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {seriesInfo.map((s) => {
              const rawVal = trendsData[s.key]?.[inspectCycleIdx];
              return (
                <div key={s.key} className="flex items-center space-x-1.5 bg-slate-950/80 px-2 py-0.5 rounded border border-[#1F2937]">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
                  <span className="text-slate-300 font-semibold">{s.meta.name}:</span>
                  <span className="text-white font-bold tabular-nums">
                    {formatSensorValue(s.key, rawVal)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SensorTrendsChart;
