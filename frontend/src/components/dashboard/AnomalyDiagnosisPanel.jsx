import React from 'react';
import {
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  Activity,
  Cpu,
  Info,
  Clock,
  CheckCircle2,
  XCircle,
  FileSearch,
} from 'lucide-react';
import { useEngine } from '../../context/EngineContext';
import { getSensorMetadata } from '../../utils/sensorMetadata';

export const AnomalyDiagnosisPanel = () => {
  const {
    selectedSubset,
    selectedUnit,
    selectedCycle,
    cycleState,
    sensorTrends,
    anomalyData,
    anomalyLoading,
  } = useEngine();

  // Extract anomaly indicators strictly from real backend detection outputs (Zero fabricated defaults)
  const hasAnomalyData = Boolean(anomalyData);
  const severity = anomalyData?.severity
    ? anomalyData.severity.toUpperCase()
    : (anomalyLoading ? 'COMPUTING...' : 'UNCHECKED');
  const isAnomaly = anomalyData?.is_anomaly ?? false;
  const ifBreached = anomalyData?.isolation_forest_breached ?? false;
  const pcaBreached = anomalyData?.pca_spe_breached ?? false;

  const ifScore = anomalyData?.isolation_forest_score ?? null;
  const pcaScore = anomalyData?.pca_spe_score ?? null;
  const tauIf = anomalyData?.thresholds?.isolation_forest ?? null;
  const tauPca = anomalyData?.thresholds?.pca_spe ?? null;

  // 3-Cycle Persistence Alarm Logic strictly from backend evaluation
  const consecutiveBreaches = anomalyData?.consecutive_breaches ?? (anomalyLoading ? null : 0);
  const persistenceAlarm = anomalyData?.persistence_alarm ?? false;

  // Extract observational evidence from current state vs. baseline cycle 1
  const stateVector = cycleState?.state_vector || {};
  const trendsData = sensorTrends?.trends || {};

  const evidenceSensors = ['sensor_3', 'sensor_7', 'sensor_4', 'sensor_9', 'sensor_11', 'sensor_15'];
  const activeEvidence = [];

  evidenceSensors.forEach((sKey) => {
    const meta = getSensorMetadata(sKey);
    if (!meta) return;

    let currVal = stateVector[meta.name] ?? stateVector[sKey];
    if (currVal === undefined && trendsData[sKey]) {
      const idx = Math.max(0, Math.min(selectedCycle - 1, (trendsData[sKey].length || 1) - 1));
      currVal = trendsData[sKey][idx];
    }
    const baseVal = trendsData[sKey]?.[0] !== undefined ? trendsData[sKey][0] : currVal;

    if (currVal !== undefined && baseVal !== undefined) {
      const delta = Number(currVal) - Number(baseVal);
      const pctDelta = baseVal !== 0 ? (delta / Math.abs(baseVal)) * 100 : 0;

      if (Math.abs(pctDelta) > 0.3) {
        activeEvidence.push({
          sensorKey: sKey,
          name: meta.name,
          station: meta.station,
          description: meta.description,
          currVal,
          baseVal,
          delta,
          pctDelta,
          units: meta.units,
          severity: Math.abs(pctDelta) > 2.0 ? 'HIGH' : 'MODERATE',
        });
      }
    }
  });

  return (
    <div className="bg-[#111827] border border-[#1F2937] rounded-xl p-5 shadow-xl shadow-black/20 space-y-5">
      {/* ========================================================================= */}
      {/* SECTION 1: Anomaly Status Header & Overall Severity Badge */}
      {/* ========================================================================= */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-[#1F2937] gap-3">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-amber-950 border border-amber-800/80 flex items-center justify-center text-amber-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white font-mono flex items-center space-x-2">
              <span>Statistical Anomaly &amp; Diagnosis Evidence</span>
              <span className="text-xs font-normal text-slate-400 font-mono">
                (Cycle {selectedCycle} Multi-Model Evaluation)
              </span>
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Independent baseline departure detection • Separated from physical fault claims
            </p>
          </div>
        </div>

        {/* Status Indicators: Severity Badge & Persistence Alarm */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Overall Severity Badge */}
          <div
            className={`px-3 py-1 rounded-md text-xs font-mono font-bold border flex items-center space-x-1.5 ${
              !hasAnomalyData
                ? 'bg-slate-900 text-slate-400 border-[#1F2937]'
                : severity === 'CRITICAL'
                ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                : severity === 'CAUTION'
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
            }`}
          >
            {severity === 'CRITICAL' ? (
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            ) : severity === 'CAUTION' ? (
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            ) : severity === 'COMPUTING...' ? (
              <Activity className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
            ) : (
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            )}
            <span>STATUS: {severity}</span>
          </div>

          {/* 3-Cycle Persistence Alarm Indicator */}
          <div
            className={`px-2.5 py-1 rounded-md text-xs font-mono border flex items-center space-x-1.5 ${
              consecutiveBreaches === null
                ? 'bg-slate-900 text-slate-400 border-[#1F2937]'
                : persistenceAlarm
                ? 'bg-rose-950/80 text-rose-300 border-rose-800/80 animate-pulse'
                : consecutiveBreaches > 0
                ? 'bg-amber-950/60 text-amber-300 border-amber-800/60'
                : 'bg-slate-900 text-slate-400 border-[#1F2937]'
            }`}
          >
            <Clock className="w-3.5 h-3.5 shrink-0" />
            <span>
              {consecutiveBreaches === null
                ? 'PERSISTENCE: COMPUTING...'
                : persistenceAlarm
                ? 'PERSISTENCE ALARM: TRIGGERED (≥3 CYCLES)'
                : consecutiveBreaches > 0
                ? `PERSISTENCE: ${consecutiveBreaches}/3 CYCLES`
                : 'PERSISTENCE: 0/3 NOMINAL'}
            </span>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2: Dual Anomaly Detector Cards (M1 & M2) */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Model 1: Regime-Aware Isolation Forest (M1) */}
        <div className="bg-slate-900/80 border border-[#1F2937] rounded-lg p-4 relative overflow-hidden">
          <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]/70">
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span className="font-mono font-bold text-sm text-white">
                Regime-Aware Isolation Forest (M1)
              </span>
            </div>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                !hasAnomalyData
                  ? 'bg-slate-800 text-slate-400 border-slate-700'
                  : ifBreached
                  ? 'bg-rose-500/10 text-rose-400 border-rose-500/20 font-bold'
                  : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              }`}
            >
              {!hasAnomalyData
                ? (anomalyLoading ? 'COMPUTING...' : 'UNCHECKED')
                : (ifBreached ? 'BREACHED' : 'NORMAL')}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 mt-3 font-mono text-xs">
            <div>
              <div className="text-[10px] uppercase text-slate-400">Decision Score</div>
              <div className="text-lg font-bold text-white tabular-nums">
                {typeof ifScore === 'number'
                  ? ifScore.toFixed(4)
                  : (anomalyLoading ? 'COMPUTING...' : '--')}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-400">Threshold (τ_IF)</div>
              <div className="text-lg font-bold text-slate-300 tabular-nums">
                {typeof tauIf === 'number'
                  ? tauIf.toFixed(4)
                  : (anomalyLoading ? '...' : '--')}
              </div>
            </div>
          </div>

          <div className="mt-2.5 pt-2 border-t border-[#1F2937]/50 text-[11px] font-mono text-slate-400">
            Evaluates high-dimensional sub-manifold density against 95th-percentile baseline boundary.
          </div>
        </div>

        {/* Model 2: PCA Squared Prediction Error (M2) */}
        <div className="bg-slate-900/80 border border-[#1F2937] rounded-lg p-4 relative overflow-hidden">
          <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]/70">
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-purple-400" />
              <span className="font-mono font-bold text-sm text-white">
                PCA Squared Prediction Error (M2)
              </span>
            </div>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                !hasAnomalyData
                  ? 'bg-slate-800 text-slate-400 border-slate-700'
                  : pcaBreached
                  ? 'bg-rose-500/10 text-rose-400 border-rose-500/20 font-bold'
                  : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              }`}
            >
              {!hasAnomalyData
                ? (anomalyLoading ? 'COMPUTING...' : 'UNCHECKED')
                : (pcaBreached ? 'BREACHED' : 'NORMAL')}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 mt-3 font-mono text-xs">
            <div>
              <div className="text-[10px] uppercase text-slate-400">Residual Energy (SPE)</div>
              <div className="text-lg font-bold text-white tabular-nums">
                {typeof pcaScore === 'number'
                  ? pcaScore.toFixed(4)
                  : (anomalyLoading ? 'COMPUTING...' : '--')}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-400">Threshold (τ_PCA)</div>
              <div className="text-lg font-bold text-slate-300 tabular-nums">
                {typeof tauPca === 'number'
                  ? tauPca.toFixed(4)
                  : (anomalyLoading ? '...' : '--')}
              </div>
            </div>
          </div>

          <div className="mt-2.5 pt-2 border-t border-[#1F2937]/50 text-[11px] font-mono text-slate-400">
            Measures aerothermal correlation breakdowns orthogonal to dominant operating subspace.
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 3: Observational Diagnostic Evidence Card */}
      {/* ========================================================================= */}
      <div className="bg-slate-900/60 border border-[#1F2937] rounded-lg p-4">
        <div className="flex items-center justify-between pb-2 border-b border-[#1F2937]/70 mb-3">
          <div className="flex items-center space-x-2">
            <FileSearch className="w-4 h-4 text-cyan-400" />
            <h4 className="font-mono font-bold text-sm text-white">
              Observational Diagnostic Evidence (Sensor Departures)
            </h4>
          </div>
          <span className="text-xs font-mono text-slate-400">
            {activeEvidence.length} Active Indicator{activeEvidence.length === 1 ? '' : 's'}
          </span>
        </div>

        {activeEvidence.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {activeEvidence.map((item) => (
              <div
                key={item.sensorKey}
                className="bg-slate-950/70 border border-[#1F2937] rounded-md p-2.5 font-mono text-xs"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-cyan-300">{item.name}</span>
                  <span
                    className={`text-[9px] px-1.5 py-0.5 rounded ${
                      item.severity === 'HIGH'
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                        : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                    }`}
                  >
                    {item.severity} DRIFT
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 truncate">{item.station}</div>
                <div className="mt-1 flex items-center justify-between text-slate-300">
                  <span>Delta:</span>
                  <span className={item.delta > 0 ? 'text-amber-400 font-bold' : 'text-cyan-400 font-bold'}>
                    {item.delta > 0 ? '+' : ''}
                    {item.delta.toFixed(2)} {item.units} ({item.pctDelta > 0 ? '+' : ''}{item.pctDelta.toFixed(1)}%)
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-4 text-center text-xs font-mono text-slate-400 flex items-center justify-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>All gas-path parameters operating within baseline nominal bounds.</span>
          </div>
        )}
      </div>

      {/* ========================================================================= */}
      {/* SECTION 4: Academic Non-Fault Identification Notice */}
      {/* ========================================================================= */}
      <div className="p-3.5 rounded-lg bg-amber-950/40 border border-amber-800/60 text-amber-300 text-xs font-mono flex items-start space-x-2.5">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-bold uppercase tracking-wider text-amber-200">
            Observational Anomaly Evidence Only • Not a Confirmed Mechanical Fault Claim
          </div>
          <p className="text-amber-300/90 leading-relaxed text-[11px]">
            OBSERVATIONAL ANOMALY EVIDENCE ONLY • NOT A CONFIRMED MECHANICAL FAULT CLAIM. Machine learning models detect statistical telemetry departures; certified fault diagnosis requires physical borescope inspection or teardown verification.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AnomalyDiagnosisPanel;
