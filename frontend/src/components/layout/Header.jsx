import React from 'react';
import { Activity, ShieldAlert, Cpu, Radio } from 'lucide-react';
import { useEngine } from '../../context/EngineContext';

export const Header = () => {
  const { backendOnline, loading } = useEngine();

  return (
    <header className="border-b border-[#1F2937] bg-[#111827]/90 backdrop-blur sticky top-0 z-50 px-4 lg:px-8 py-3">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        {/* Left: Branding & Operational Title */}
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-950 border border-cyan-800/80 flex items-center justify-center text-cyan-400 shadow-sm shadow-cyan-950/50">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-semibold tracking-wide text-white">
                Aircraft Engine Digital Twin & AI Prognostics
              </h1>
              <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono font-medium">
                C-MAPSS FD001-FD004
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono">
              Aerospace Operations Control Center • Real-Time Digital Twin Telemetry
            </p>
          </div>
        </div>

        {/* Right: Academic Disclaimer & Online Status */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Academic Non-Authority Disclaimer Pill */}
          <div className="flex items-center space-x-2 px-2.5 py-1 rounded-md bg-amber-950/40 border border-amber-800/50 text-amber-300 text-xs font-mono">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span className="truncate max-w-xs md:max-w-md">
              ACADEMIC SIMULATION ONLY • NOT FOR FLIGHT CONTROL OR MAINTENANCE DISPATCH
            </span>
          </div>

          {/* Backend Connection Indicator */}
          <div className="flex items-center space-x-2 px-2.5 py-1 rounded-md bg-slate-900 border border-[#1F2937] text-xs font-mono">
            <Radio className={`w-3.5 h-3.5 ${backendOnline ? 'text-emerald-400 animate-pulse' : 'text-rose-500'}`} />
            <span className="text-slate-300">API:</span>
            <span className={`font-semibold ${backendOnline ? 'text-emerald-400' : 'text-rose-400'}`}>
              {backendOnline ? (loading ? 'SYNCING' : 'ONLINE') : 'OFFLINE'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
