import React, { useState, useMemo } from 'react';
import { useEngine } from '../../context/EngineContext';
import { formatSensorValue, getSensorMetadata } from '../../utils/sensorMetadata';

/**
 * Station Definitions for the Turbofan Cutaway Schematic
 */
const ENGINE_STATIONS = [
  {
    id: 'station_2',
    name: 'Station 2 - Fan Inlet',
    shortName: 'Station 2 (Fan Inlet)',
    x: 95,
    y: 110,
    sensors: ['sensor_1', 'sensor_5'], // T2, P2
    description: 'Inlet total temperature and pressure entering the engine fan stage.',
  },
  {
    id: 'station_15',
    name: 'Station 1.5 - Bypass Duct',
    shortName: 'Station 1.5 (Bypass Duct)',
    x: 320,
    y: 35,
    sensors: ['sensor_6', 'sensor_15'], // P15, BPR
    description: 'Bypass duct airflow separating from core flow into outer annular duct.',
  },
  {
    id: 'station_24',
    name: 'Station 2.4 - LPC Outlet',
    shortName: 'Station 2.4 (LPC Outlet)',
    x: 215,
    y: 110,
    sensors: ['sensor_2'], // T24
    description: 'Intermediate compressor exit entering the high-pressure core compressor.',
  },
  {
    id: 'station_3',
    name: 'Station 3 - HPC Outlet',
    shortName: 'Station 3 (HPC Outlet)',
    x: 360,
    y: 110,
    sensors: ['sensor_3', 'sensor_7', 'sensor_11'], // T30, P30, Ps30
    derived: ['hpc_temp_rise', 'overall_pressure_ratio'],
    description: 'High-Pressure Compressor discharge directly prior to combustor injection.',
  },
  {
    id: 'station_4',
    name: 'Station 4 - Combustor / Burner',
    shortName: 'Station 4 (Combustor)',
    x: 460,
    y: 110,
    sensors: ['sensor_12', 'sensor_16'], // phi, farB
    description: 'Primary combustion zone where fuel-air mixture ignites at peak cycle temperature.',
  },
  {
    id: 'station_45',
    name: 'Station 4.5 - HPT Core Spool',
    shortName: 'Station 4.5 (HPT Core)',
    x: 550,
    y: 110,
    sensors: ['sensor_9', 'sensor_14', 'sensor_20'], // Nc, NRc, W31
    description: 'High-Pressure Turbine extracting energy to drive the core compressor spool.',
  },
  {
    id: 'station_5',
    name: 'Station 5 - LPT Outlet / EGT',
    shortName: 'Station 5 (LPT / EGT)',
    x: 655,
    y: 110,
    sensors: ['sensor_4', 'sensor_8', 'sensor_13', 'sensor_21'], // T50, Nf, NRf, W32
    derived: ['spool_speed_ratio'],
    description: 'Low-Pressure Turbine discharge and Exhaust Gas Temperature (EGT) station.',
  },
];

/**
 * 40-Variable State Vector Canonical Taxonomy Definition
 */
const STATE_TAXONOMY_DEF = [
  // G1 - G5 Operational Metadata & Environmental Settings
  { g: 'G1', key: 'unit_number', name: 'Engine Unit ID', unit: '--', classification: 'MEASURED' },
  { g: 'G2', key: 'time_cycles', name: 'Operating Flight Cycle', unit: 'cycles', classification: 'MEASURED' },
  { g: 'G3', key: 'altitude', name: 'Pressure Altitude', unit: 'kft', classification: 'MEASURED' },
  { g: 'G4', key: 'mach_number', name: 'Mach Number', unit: 'Mach', classification: 'MEASURED' },
  { g: 'G5', key: 'throttle_resolver_angle', name: 'Throttle Resolver Angle', unit: '% TRA', classification: 'MEASURED' },
  // G6 - G26 Gas-Path Thermodynamic Sensors
  { g: 'G6', key: 'T2', name: 'Fan Inlet Total Temp (T2)', unit: '°R', classification: 'MEASURED', sensorKey: 'sensor_1' },
  { g: 'G7', key: 'T24', name: 'LPC Outlet Total Temp (T24)', unit: '°R', classification: 'MEASURED', sensorKey: 'sensor_2' },
  { g: 'G8', key: 'T30', name: 'HPC Outlet Total Temp (T30)', unit: '°R', classification: 'MEASURED', sensorKey: 'sensor_3' },
  { g: 'G9', key: 'T50', name: 'LPT Outlet Total Temp (T50)', unit: '°R', classification: 'MEASURED', sensorKey: 'sensor_4' },
  { g: 'G10', key: 'P2', name: 'Fan Inlet Total Press (P2)', unit: 'psia', classification: 'MEASURED', sensorKey: 'sensor_5' },
  { g: 'G11', key: 'P15', name: 'Bypass Duct Total Press (P15)', unit: 'psia', classification: 'MEASURED', sensorKey: 'sensor_6' },
  { g: 'G12', key: 'P30', name: 'HPC Outlet Total Press (P30)', unit: 'psia', classification: 'MEASURED', sensorKey: 'sensor_7' },
  { g: 'G13', key: 'Nf', name: 'Physical Fan Speed (Nf)', unit: 'rpm', classification: 'MEASURED', sensorKey: 'sensor_8' },
  { g: 'G14', key: 'Nc', name: 'Physical Core Speed (Nc)', unit: 'rpm', classification: 'MEASURED', sensorKey: 'sensor_9' },
  { g: 'G15', key: 'epr', name: 'Engine Pressure Ratio (EPR)', unit: '--', classification: 'MEASURED', sensorKey: 'sensor_10' },
  { g: 'G16', key: 'Ps30', name: 'HPC Static Pressure (Ps30)', unit: 'psia', classification: 'MEASURED', sensorKey: 'sensor_11' },
  { g: 'G17', key: 'phi', name: 'Fuel to Ps30 Ratio (phi)', unit: 'pps/psia', classification: 'MEASURED', sensorKey: 'sensor_12' },
  { g: 'G18', key: 'NRf', name: 'Corrected Fan Speed (NRf)', unit: 'rpm', classification: 'MEASURED', sensorKey: 'sensor_13' },
  { g: 'G19', key: 'NRc', name: 'Corrected Core Speed (NRc)', unit: 'rpm', classification: 'MEASURED', sensorKey: 'sensor_14' },
  { g: 'G20', key: 'BPR', name: 'Bypass Ratio (BPR)', unit: '--', classification: 'MEASURED', sensorKey: 'sensor_15' },
  { g: 'G21', key: 'farB', name: 'Burner Fuel-Air Ratio (farB)', unit: '--', classification: 'MEASURED', sensorKey: 'sensor_16' },
  { g: 'G22', key: 'htBleed', name: 'Bleed Enthalpy (htBleed)', unit: 'lbf', classification: 'MEASURED', sensorKey: 'sensor_17' },
  { g: 'G23', key: 'Nf_dmd', name: 'Demanded Fan Speed (Nf_dmd)', unit: 'rpm', classification: 'MEASURED', sensorKey: 'sensor_18' },
  { g: 'G24', key: 'PCNfR_dmd', name: 'Demanded Corrected Speed', unit: '%', classification: 'MEASURED', sensorKey: 'sensor_19' },
  { g: 'G25', key: 'W31', name: 'HPT Coolant Bleed (W31)', unit: 'lbm/s', classification: 'MEASURED', sensorKey: 'sensor_20' },
  { g: 'G26', key: 'W32', name: 'LPT Coolant Bleed (W32)', unit: 'lbm/s', classification: 'MEASURED', sensorKey: 'sensor_21' },
  // G27 - G31 Physical Derived Indicators
  { g: 'G27', key: 'hpc_temp_rise', name: 'HPC Temperature Rise (ΔT)', unit: '°R', classification: 'DERIVED' },
  { g: 'G28', key: 'overall_pressure_ratio', name: 'Overall Pressure Ratio (OPR)', unit: '--', classification: 'DERIVED' },
  { g: 'G29', key: 'spool_speed_ratio', name: 'Spool Speed Ratio (Nc/Nf)', unit: '--', classification: 'DERIVED' },
  { g: 'G30', key: 'normalized_feature_vec', name: 'SafeMinMax Scaled Vector', unit: '[0, 1]', classification: 'DERIVED' },
  { g: 'G31', key: 'temporal_seq_tensor', name: 'Temporal Sequence Window (30, D)', unit: 'tensor', classification: 'DERIVED' },
  // G32 - G36 Machine Learning Predicted States
  { g: 'G32', key: 'anomaly_score_m1', name: 'Isolation Forest Anomaly Score', unit: 'score', classification: 'PREDICTED' },
  { g: 'G33', key: 'anomaly_score_m2', name: 'PCA-SPE Residual (Q-stat)', unit: 'residual', classification: 'PREDICTED' },
  { g: 'G34', key: 'anomaly_alert_flag', name: 'Persistence Alert Alarm', unit: 'flag {0,1}', classification: 'PREDICTED' },
  { g: 'G35', key: 'predicted_rul_xgboost', name: 'XGBoost Predicted RUL', unit: 'cycles', classification: 'PREDICTED' },
  { g: 'G36', key: 'predicted_rul_lstm', name: 'Bi-LSTM Predicted RUL', unit: 'cycles', classification: 'PREDICTED' },
  // G37 - G38 Derived Diagnostics & Health State
  { g: 'G37', key: 'prognostic_spread', name: 'Prognostic Spread (|XGB - LSTM|)', unit: 'cycles', classification: 'DERIVED' },
  { g: 'G38', key: 'virtual_health_index', name: 'Virtual Engine Health Index (VHI)', unit: '[0, 1]', classification: 'DERIVED' },
  // G39 - G40 Simulated What-If Perturbation States
  { g: 'G39', key: 'thermal_stress_offset', name: 'Thermal Stress Offset (ΔT_sim)', unit: '°R', classification: 'SIMULATED' },
  { g: 'G40', key: 'sensor_bias_delta', name: 'Sensor Bias Injection Delta', unit: 'bias delta', classification: 'SIMULATED' },
];

export const DigitalTwinSimulationPanel = () => {
  const {
    selectedSubset,
    selectedUnit,
    selectedCycle,
    cycleState,
    simulationResult,
    simulationLoading,
    runSimulation,
    clearSimulation,
  } = useEngine();

  // Active Station in Cutaway Schematic
  const [activeStationId, setActiveStationId] = useState('station_3');

  // Provenance Filter: 'ALL' | 'MEASURED' | 'DERIVED' | 'PREDICTED' | 'SIMULATED'
  const [provenanceFilter, setProvenanceFilter] = useState('ALL');

  // What-If Simulation Controls
  const [selectedScenario, setSelectedScenario] = useState('THERMAL_PERTURBATION');
  const [targetVariable, setTargetVariable] = useState('thermal_stress_offset');
  const [proposedValue, setProposedValue] = useState(4.0);
  const [operatorApproval, setOperatorApproval] = useState(false);
  const [simulationError, setSimulationError] = useState(null);

  // Active station data
  const activeStation = useMemo(() => {
    return ENGINE_STATIONS.find((st) => st.id === activeStationId) || ENGINE_STATIONS[0];
  }, [activeStationId]);

  // Handle Scenario Switch
  const handleScenarioChange = (e) => {
    const scenario = e.target.value;
    setSelectedScenario(scenario);
    if (scenario === 'THERMAL_PERTURBATION') {
      setTargetVariable('thermal_stress_offset');
      setProposedValue(4.0);
    } else if (scenario === 'SENSOR_BIAS_INJECTION') {
      setTargetVariable('sensor_bias_delta');
      setProposedValue(5.0);
    }
  };

  // Execute What-If Simulation
  const handleExecuteSimulation = async () => {
    setSimulationError(null);
    try {
      await runSimulation({
        subset: selectedSubset,
        unit_number: selectedUnit,
        cycle: selectedCycle,
        action_type: selectedScenario,
        target_variable: targetVariable,
        proposed_value: parseFloat(proposedValue),
        operator_approval: Boolean(operatorApproval),
      });
    } catch (err) {
      console.error('Failed to execute simulation:', err);
      setSimulationError(err.response?.data?.detail || err.message || 'Simulation execution failed');
    }
  };

  // Helper for rendering badges
  const renderProvenanceBadge = (classification, size = 'sm') => {
    const pad = size === 'xs' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-0.5 text-xs';
    switch (classification) {
      case 'MEASURED':
        return (
          <span className={`inline-flex items-center gap-1 font-mono font-medium rounded border bg-blue-950/60 text-blue-400 border-blue-700/60 ${pad}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
            MEASURED / DATASET
          </span>
        );
      case 'DERIVED':
        return (
          <span className={`inline-flex items-center gap-1 font-mono font-medium rounded border bg-emerald-950/60 text-emerald-400 border-emerald-700/60 ${pad}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            PHYSICAL / DERIVED
          </span>
        );
      case 'PREDICTED':
        return (
          <span className={`inline-flex items-center gap-1 font-mono font-medium rounded border bg-purple-950/60 text-purple-400 border-purple-700/60 ${pad}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
            MODEL / PREDICTED
          </span>
        );
      case 'SIMULATED':
        return (
          <span className={`inline-flex items-center gap-1 font-mono font-medium rounded border bg-amber-950/60 text-amber-400 border-amber-600/70 ${pad}`}>
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
            SIMULATED / SCENARIO
          </span>
        );
      default:
        return (
          <span className={`inline-flex items-center font-mono rounded border bg-slate-800 text-slate-300 border-slate-700 ${pad}`}>
            {classification}
          </span>
        );
    }
  };

  // Helper to extract state values from cycleState
  const getStateValue = (def) => {
    if (!cycleState?.state_vector) return null;
    const sv = cycleState.state_vector;
    if (sv[def.g] !== undefined) return sv[def.g];
    if (sv[def.key] !== undefined) return sv[def.key];
    if (sv[`${def.g}_${def.key}`] !== undefined) return sv[`${def.g}_${def.key}`];
    return null;
  };

  // Filtered state vector list
  const filteredStateTaxonomy = useMemo(() => {
    if (provenanceFilter === 'ALL') return STATE_TAXONOMY_DEF;
    return STATE_TAXONOMY_DEF.filter((item) => item.classification === provenanceFilter);
  }, [provenanceFilter]);

  return (
    <div className="bg-[#111827] border border-[#1F2937] rounded-xl p-5 shadow-lg space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-[#1F2937] gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            <h2 className="text-lg font-semibold tracking-wide text-white uppercase">
              Digital Twin State &amp; What-If Simulation Sandbox
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Aerothermal station topology, 40-dimensional state vector provenance, and closed-loop What-If parameter perturbation.
          </p>
        </div>

        {/* Provenance Distinction Legend */}
        <div className="flex flex-wrap items-center gap-2 bg-[#0B0F19] px-3 py-1.5 rounded-lg border border-[#1F2937]">
          <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mr-1">Provenance:</span>
          {renderProvenanceBadge('MEASURED', 'xs')}
          {renderProvenanceBadge('DERIVED', 'xs')}
          {renderProvenanceBadge('PREDICTED', 'xs')}
          {renderProvenanceBadge('SIMULATED', 'xs')}
        </div>
      </div>

      {/* Section 1: Interactive SVG Turbofan Engine Cutaway Schematic */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
            <span>Section 1: Turbofan Engine Cutaway Schematic &amp; Gas-Path Stations</span>
          </h3>
          <span className="text-xs font-mono text-cyan-400">
            Selected Station: {activeStation.shortName}
          </span>
        </div>

        {/* Interactive SVG Engine Schematic Canvas */}
        <div className="relative bg-[#070B14] rounded-lg border border-[#1F2937] p-4 overflow-x-auto">
          <svg
            viewBox="0 0 760 220"
            className="w-full h-auto min-w-[700px] select-none"
            aria-label="Turbofan Cutaway Schematic"
          >
            <defs>
              <linearGradient id="casingGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#1E293B" stopOpacity="0.8" />
                <stop offset="50%" stopColor="#334155" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#1E293B" stopOpacity="0.8" />
              </linearGradient>
              <linearGradient id="coreGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#0F172A" />
                <stop offset="50%" stopColor="#1E293B" />
                <stop offset="100%" stopColor="#0F172A" />
              </linearGradient>
              <linearGradient id="combustorFlame" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#F97316" stopOpacity="0.6" />
                <stop offset="50%" stopColor="#EF4444" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#EA580C" stopOpacity="0.6" />
              </linearGradient>
            </defs>

            {/* Nacelle & Outer Bypass Casing */}
            <path
              d="M 50 40 Q 220 20 420 20 Q 640 20 700 45 L 680 75 Q 450 55 240 55 L 75 75 Z"
              fill="url(#casingGrad)"
              stroke="#475569"
              strokeWidth="1.5"
            />
            <path
              d="M 50 180 Q 220 200 420 200 Q 640 200 700 175 L 680 145 Q 450 165 240 165 L 75 145 Z"
              fill="url(#casingGrad)"
              stroke="#475569"
              strokeWidth="1.5"
            />

            {/* Bypass Duct Airflow Indicator Arrows */}
            <g stroke="#38BDF8" strokeWidth="1.5" strokeDasharray="4,4" fill="none" opacity="0.6">
              <path d="M 120 48 Q 320 35 550 42" />
              <path d="M 120 172 Q 320 185 550 178" />
            </g>
            <text x="320" y="48" fill="#38BDF8" fontSize="10" fontFamily="monospace" textAnchor="middle">
              BYPASS DUCT AIRFLOW (Station 1.5)
            </text>

            {/* Engine Core Spool Structure */}
            <rect x="180" y="80" width="510" height="60" rx="4" fill="url(#coreGrad)" stroke="#334155" strokeWidth="1.5" />

            {/* Fan Stage (Station 2) */}
            <g id="fan-stage">
              <ellipse cx="95" cy="110" rx="20" ry="60" fill="#1E293B" stroke="#0EA5E9" strokeWidth="2" />
              <line x1="95" y1="52" x2="95" y2="168" stroke="#38BDF8" strokeWidth="3" />
              <path d="M 78 80 L 112 140 M 78 140 L 112 80" stroke="#0284C7" strokeWidth="2" />
              <circle cx="95" cy="110" r="14" fill="#0369A1" stroke="#38BDF8" strokeWidth="1.5" />
            </g>

            {/* LPC Stage (Station 2.4) */}
            <g id="lpc-stage">
              <polygon points="195,90 235,95 235,125 195,130" fill="#1E293B" stroke="#60A5FA" strokeWidth="1.5" />
              <line x1="215" y1="92" x2="215" y2="128" stroke="#3B82F6" strokeWidth="2" />
            </g>

            {/* HPC Stage (Station 3) */}
            <g id="hpc-stage">
              <polygon points="250,88 370,96 370,124 250,132" fill="#1E293B" stroke="#10B981" strokeWidth="1.5" />
              <line x1="280" y1="90" x2="280" y2="130" stroke="#059669" strokeWidth="1.5" />
              <line x1="310" y1="92" x2="310" y2="128" stroke="#059669" strokeWidth="1.5" />
              <line x1="340" y1="94" x2="340" y2="126" stroke="#059669" strokeWidth="1.5" />
            </g>

            {/* Combustor Chamber (Station 4) */}
            <g id="combustor-stage">
              <rect x="385" y="86" width="95" height="48" rx="6" fill="url(#combustorFlame)" stroke="#DC2626" strokeWidth="1.5" />
              <text x="432" y="114" fill="#FEF08A" fontSize="9" fontFamily="monospace" fontWeight="bold" textAnchor="middle">
                BURNER
              </text>
            </g>

            {/* HPT Stage (Station 4.5) */}
            <g id="hpt-stage">
              <polygon points="495,96 545,91 545,129 495,124" fill="#1E293B" stroke="#F59E0B" strokeWidth="1.5" />
              <line x1="520" y1="93" x2="520" y2="127" stroke="#D97706" strokeWidth="2" />
            </g>

            {/* LPT Stage & Core Exhaust (Station 5) */}
            <g id="lpt-stage">
              <polygon points="560,92 660,86 660,134 560,128" fill="#1E293B" stroke="#EC4899" strokeWidth="1.5" />
              <line x1="590" y1="90" x2="590" y2="130" stroke="#DB2777" strokeWidth="1.5" />
              <line x1="625" y1="88" x2="625" y2="132" stroke="#DB2777" strokeWidth="1.5" />
            </g>

            {/* Core Exhaust Plug / Cone */}
            <polygon points="660,98 710,110 660,122" fill="#334155" stroke="#64748B" strokeWidth="1.5" />

            {/* Station Hotspot Markers (Clickable) */}
            {ENGINE_STATIONS.map((station) => {
              const isSelected = station.id === activeStationId;
              return (
                <g
                  key={station.id}
                  onClick={() => setActiveStationId(station.id)}
                  className="cursor-pointer group"
                >
                  {/* Glowing hotspot halo if active */}
                  {isSelected && (
                    <circle
                      cx={station.x}
                      cy={station.y}
                      r="16"
                      fill="#06B6D4"
                      fillOpacity="0.25"
                      stroke="#06B6D4"
                      strokeWidth="1.5"
                      className="animate-pulse"
                    />
                  )}
                  {/* Pin Dot */}
                  <circle
                    cx={station.x}
                    cy={station.y}
                    r="8"
                    fill={isSelected ? '#06B6D4' : '#1E293B'}
                    stroke={isSelected ? '#FFFFFF' : '#38BDF8'}
                    strokeWidth="2"
                    className="transition-all duration-200 group-hover:scale-125"
                  />
                  {/* Station Label Callout */}
                  <rect
                    x={station.x - 30}
                    y={station.y > 100 ? station.y + 14 : station.y - 28}
                    width="60"
                    height="16"
                    rx="3"
                    fill="#0F172A"
                    stroke={isSelected ? '#06B6D4' : '#334155'}
                    strokeWidth="1"
                  />
                  <text
                    x={station.x}
                    y={station.y > 100 ? station.y + 26 : station.y - 16}
                    fill={isSelected ? '#38BDF8' : '#94A3B8'}
                    fontSize="9"
                    fontFamily="monospace"
                    fontWeight="bold"
                    textAnchor="middle"
                  >
                    {station.id.replace('station_', 'ST-')}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Interactive Station Quick Readout Drawer */}
          <div className="mt-3 p-3 bg-[#0B0F19] rounded-lg border border-[#1F2937] flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-bold text-cyan-400 uppercase">
                  {activeStation.name}
                </span>
                {renderProvenanceBadge('MEASURED', 'xs')}
              </div>
              <p className="text-xs text-slate-400 mt-0.5">{activeStation.description}</p>
            </div>

            {/* Station Sensors & Values */}
            <div className="flex flex-wrap items-center gap-3">
              {activeStation.sensors.map((sKey) => {
                const meta = getSensorMetadata(sKey);
                const rawVal = cycleState?.state_vector?.[sKey] ?? cycleState?.state_vector?.[meta?.name];
                return (
                  <div
                    key={sKey}
                    className="bg-[#111827] border border-[#1F2937] px-3 py-1.5 rounded flex items-center gap-2"
                  >
                    <span className="text-xs font-mono text-slate-400">{meta?.name || sKey}:</span>
                    <span className="text-xs font-mono font-semibold text-white">
                      {formatSensorValue(sKey, rawVal)}
                    </span>
                    <span className="text-[10px] font-mono text-blue-400 bg-blue-950/60 px-1 py-0.2 rounded border border-blue-800/50">
                      DATASET
                    </span>
                  </div>
                );
              })}

              {/* Station Derived Values if present */}
              {activeStation.derived?.map((dKey) => {
                const def = STATE_TAXONOMY_DEF.find((item) => item.key === dKey);
                const val = getStateValue(def || { key: dKey, g: '' });
                return (
                  <div
                    key={dKey}
                    className="bg-[#111827] border border-[#1F2937] px-3 py-1.5 rounded flex items-center gap-2"
                  >
                    <span className="text-xs font-mono text-slate-400">{def?.name || dKey}:</span>
                    <span className="text-xs font-mono font-semibold text-emerald-400">
                      {val !== null && val !== undefined ? Number(val).toFixed(2) : '--'}{' '}
                      <span className="text-[10px] text-slate-500">{def?.unit}</span>
                    </span>
                    <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/60 px-1 py-0.2 rounded border border-emerald-800/50">
                      DERIVED
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Section 2: Complete 40-Dimensional State Vector Provenance Breakdown */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <h3 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
            <span>Section 2: 40-Variable State Vector (G1 – G40) &amp; Provenance Classification</span>
          </h3>

          {/* Filter Pills */}
          <div className="flex flex-wrap items-center gap-1.5 text-xs font-mono">
            {['ALL', 'MEASURED', 'DERIVED', 'PREDICTED', 'SIMULATED'].map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setProvenanceFilter(cat)}
                className={`px-2.5 py-1 rounded transition-colors ${
                  provenanceFilter === cat
                    ? 'bg-cyan-600 text-white font-bold'
                    : 'bg-[#0B0F19] text-slate-400 hover:text-slate-200 border border-[#1F2937]'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Scrollable State Vector Grid */}
        <div className="max-h-72 overflow-y-auto rounded-lg border border-[#1F2937] bg-[#070B14] p-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
            {filteredStateTaxonomy.map((item) => {
              const val = getStateValue(item);
              const isSimulated = item.classification === 'SIMULATED';
              // Check if simulation perturbed this variable
              const simAfterVal = simulationResult?.state_after?.[item.key] ?? simulationResult?.state_after?.[item.g];
              const displayVal = simAfterVal !== undefined ? simAfterVal : val;

              return (
                <div
                  key={item.g}
                  className={`p-2.5 rounded-lg border transition-all ${
                    isSimulated
                      ? 'bg-amber-950/20 border-amber-800/50'
                      : 'bg-[#111827] border-[#1F2937]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono font-bold text-cyan-400">{item.g}</span>
                    {renderProvenanceBadge(item.classification, 'xs')}
                  </div>
                  <div className="text-xs font-medium text-slate-300 truncate" title={item.name}>
                    {item.name}
                  </div>
                  <div className="mt-1 flex items-baseline justify-between">
                    <span className="text-sm font-mono font-semibold text-white">
                      {displayVal !== null && displayVal !== undefined
                        ? typeof displayVal === 'number'
                          ? Number(displayVal).toFixed(2)
                          : String(displayVal)
                        : '--'}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">{item.unit}</span>
                  </div>
                  {simAfterVal !== undefined && (
                    <div className="mt-1 text-[10px] font-mono text-amber-400 flex items-center gap-1">
                      <span>Perturbed Candidate Value</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Section 3: Interactive What-If Scenario Sandbox */}
      <div className="space-y-4 pt-2 border-t border-[#1F2937]">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
            <span>Section 3: Interactive What-If Scenario Sandbox (Closed-Loop Safety Gated)</span>
          </h3>
          <span className="text-xs font-mono text-amber-400 bg-amber-950/50 px-2 py-0.5 rounded border border-amber-800/60">
            Phase 5 Certified Scenarios Only
          </span>
        </div>

        {/* Sandbox Form Card */}
        <div className="bg-[#0B0F19] rounded-lg border border-[#1F2937] p-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Scenario Selection */}
            <div>
              <label className="block text-xs font-mono text-slate-300 uppercase mb-1">
                Scenario Mode (Phase 5 Supported)
              </label>
              <select
                value={selectedScenario}
                onChange={handleScenarioChange}
                className="w-full bg-[#111827] border border-[#1F2937] text-white text-xs rounded-lg px-3 py-2 font-mono focus:outline-none focus:border-cyan-500"
              >
                <option value="THERMAL_PERTURBATION">Thermal Stress Perturbation (G39)</option>
                <option value="SENSOR_BIAS_INJECTION">Sensor Bias Injection (G40)</option>
              </select>
              <p className="text-[11px] text-slate-400 mt-1">
                {selectedScenario === 'THERMAL_PERTURBATION'
                  ? 'Inject hypothetical gas-path thermal stress offset to evaluate aerothermal safety margin.'
                  : 'Inject hypothetical sensor bias drift into instrumentation telemetry channels.'}
              </p>
            </div>

            {/* Target Variable Selection */}
            <div>
              <label className="block text-xs font-mono text-slate-300 uppercase mb-1">
                Target Perturbation Channel
              </label>
              <select
                value={targetVariable}
                onChange={(e) => setTargetVariable(e.target.value)}
                className="w-full bg-[#111827] border border-[#1F2937] text-white text-xs rounded-lg px-3 py-2 font-mono focus:outline-none focus:border-cyan-500"
              >
                {selectedScenario === 'THERMAL_PERTURBATION' ? (
                  <>
                    <option value="thermal_stress_offset">thermal_stress_offset (G39 ΔT)</option>
                    <option value="T30">T30 (HPC Outlet Total Temp)</option>
                    <option value="T50">T50 (LPT Outlet EGT)</option>
                  </>
                ) : (
                  <>
                    <option value="sensor_bias_delta">sensor_bias_delta (G40)</option>
                    <option value="T30">s_3 / T30 (HPC Temp Bias)</option>
                    <option value="T24">s_2 / T24 (LPC Temp Bias)</option>
                    <option value="T50">s_4 / T50 (LPT EGT Bias)</option>
                    <option value="P30">s_7 / P30 (HPC Pressure Bias)</option>
                    <option value="NRc">s_14 / NRc (Core Speed Bias)</option>
                  </>
                )}
              </select>
              <span className="text-[11px] font-mono text-amber-400 mt-1 inline-block">
                Classification: SIMULATED (Synthetic state injection)
              </span>
            </div>

            {/* Proposed Perturbation Value Slider & Input */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-mono text-slate-300 uppercase">
                  Perturbation Magnitude:
                </label>
                <span className="text-xs font-mono font-bold text-amber-400">
                  {proposedValue > 0 ? `+${proposedValue}` : proposedValue}{' '}
                  {selectedScenario === 'THERMAL_PERTURBATION' ? '°R' : 'units'}
                </span>
              </div>
              <input
                type="range"
                min={selectedScenario === 'THERMAL_PERTURBATION' ? -5.0 : -20.0}
                max={selectedScenario === 'THERMAL_PERTURBATION' ? 30.0 : 20.0}
                step="0.5"
                value={proposedValue}
                onChange={(e) => setProposedValue(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
                <span>{selectedScenario === 'THERMAL_PERTURBATION' ? '-5.0 °R' : '-20.0'}</span>
                <span>0.0</span>
                <span>{selectedScenario === 'THERMAL_PERTURBATION' ? '+30.0 °R' : '+20.0'}</span>
              </div>
            </div>
          </div>

          {/* Operator Signoff Checkbox & Execution Buttons */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-3 border-t border-[#1F2937]">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={operatorApproval}
                onChange={(e) => setOperatorApproval(e.target.checked)}
                className="w-4 h-4 rounded bg-[#111827] border-slate-600 text-cyan-600 focus:ring-0 focus:ring-offset-0 cursor-pointer"
              />
              <span className="text-xs text-slate-300">
                Operator Signoff: Authorize high-impact candidate state update if required by safety governor
              </span>
            </label>

            <div className="flex items-center gap-2">
              {simulationResult && (
                <button
                  type="button"
                  onClick={clearSimulation}
                  disabled={simulationLoading}
                  className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800 text-slate-300 text-xs font-mono hover:bg-slate-700 transition-colors"
                >
                  Reset Simulation
                </button>
              )}
              <button
                type="button"
                onClick={handleExecuteSimulation}
                disabled={simulationLoading}
                className="px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-mono font-semibold transition-colors flex items-center gap-2 shadow-md shadow-cyan-900/30"
              >
                {simulationLoading ? (
                  <>
                    <svg className="animate-spin h-3.5 w-3.5 text-white" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                    <span>Verifying Envelope...</span>
                  </>
                ) : (
                  <span>Simulate Candidate State</span>
                )}
              </button>
            </div>
          </div>

          {/* Simulation Error Alert if present */}
          {simulationError && (
            <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 font-mono">
              Error executing What-If simulation: {simulationError}
            </div>
          )}
        </div>
      </div>

      {/* Section 4: Baseline vs. Simulated Output Comparison Matrix */}
      {simulationResult && (
        <div className="space-y-4 pt-2 border-t border-[#1F2937]">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
              <span>Section 4: Baseline vs. Simulated Candidate State Comparison</span>
            </h3>
            <span className="text-xs font-mono text-slate-400">
              Action ID: <span className="text-cyan-400">{simulationResult.action_id}</span>
            </span>
          </div>

          {/* Closed-Loop Safety Governor Decision Banner */}
          <div
            className={`p-3.5 rounded-lg border flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
              simulationResult.decision_status === 'ACCEPTED'
                ? 'bg-emerald-950/30 border-emerald-700/60'
                : simulationResult.decision_status === 'REQUIRES_REVIEW'
                ? 'bg-amber-950/30 border-amber-700/60'
                : 'bg-rose-950/30 border-rose-700/60'
            }`}
          >
            <div className="flex items-center gap-3">
              <span
                className={`px-2.5 py-1 rounded font-mono font-bold text-xs border ${
                  simulationResult.decision_status === 'ACCEPTED'
                    ? 'bg-emerald-900 text-emerald-200 border-emerald-500'
                    : simulationResult.decision_status === 'REQUIRES_REVIEW'
                    ? 'bg-amber-900 text-amber-200 border-amber-500'
                    : 'bg-rose-900 text-rose-200 border-rose-500'
                }`}
              >
                GOVERNOR DECISION: {simulationResult.decision_status}
              </span>
              <span className="text-xs text-slate-300">{simulationResult.reason_summary}</span>
            </div>

            <div className="text-xs font-mono text-slate-400">
              State Updated:{' '}
              <span className={simulationResult.update_applied ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                {simulationResult.update_applied ? 'YES (APPLIED)' : 'NO (BLOCKED)'}
              </span>
            </div>
          </div>

          {/* Side-by-Side Telemetry Delta Comparison Matrix */}
          <div className="overflow-x-auto rounded-lg border border-[#1F2937]">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#0B0F19] text-slate-400 uppercase text-[11px] border-b border-[#1F2937]">
                <tr>
                  <th className="py-2.5 px-3">State Channel</th>
                  <th className="py-2.5 px-3">
                    <span className="text-blue-400">Baseline Measured Value</span>
                  </th>
                  <th className="py-2.5 px-3">
                    <span className="text-amber-400">Simulated Candidate Value</span>
                  </th>
                  <th className="py-2.5 px-3">Delta (Δ)</th>
                  <th className="py-2.5 px-3">Safety Verification Envelope</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1F2937] bg-[#070B14]">
                {/* 1. Target Perturbation Variable */}
                <tr className="hover:bg-[#111827]/50">
                  <td className="py-2 px-3 font-semibold text-white">
                    {targetVariable}
                    <span className="ml-2 text-[10px] text-amber-400 font-normal">
                      (Target Perturbation)
                    </span>
                  </td>
                  <td className="py-2 px-3">
                    <div className="flex items-center gap-1.5">
                      <span className="font-semibold text-slate-200">
                        {simulationResult.state_before?.[targetVariable] !== undefined
                          ? Number(simulationResult.state_before[targetVariable]).toFixed(2)
                          : '0.00'}
                      </span>
                      {renderProvenanceBadge('MEASURED', 'xs')}
                    </div>
                  </td>
                  <td className="py-2 px-3">
                    <div className="flex items-center gap-1.5">
                      <span className="font-semibold text-amber-400">
                        {simulationResult.state_after?.[targetVariable] !== undefined
                          ? Number(simulationResult.state_after[targetVariable]).toFixed(2)
                          : Number(proposedValue).toFixed(2)}
                      </span>
                      {renderProvenanceBadge('SIMULATED', 'xs')}
                    </div>
                  </td>
                  <td className="py-2 px-3 font-bold text-amber-300">
                    {proposedValue > 0 ? `+${proposedValue}` : proposedValue}
                  </td>
                  <td className="py-2 px-3">
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded border ${
                        simulationResult.decision_status === 'ACCEPTED'
                          ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                          : 'bg-rose-950 text-rose-400 border-rose-800'
                      }`}
                    >
                      {simulationResult.decision_status === 'ACCEPTED' ? 'SAFE TOLERANCE' : 'EXCEEDED'}
                    </span>
                  </td>
                </tr>

                {/* 2. Key Gas Path Channels impacted (T30, T50, P30, OPR) */}
                {['T30', 'T50', 'P30', 'hpc_temp_rise'].map((channel) => {
                  const beforeVal = simulationResult.state_before?.[channel];
                  const afterVal = simulationResult.state_after?.[channel];
                  if (beforeVal === undefined && afterVal === undefined) return null;
                  const delta =
                    beforeVal !== undefined && afterVal !== undefined ? afterVal - beforeVal : 0;
                  const meta = getSensorMetadata(channel);

                  return (
                    <tr key={channel} className="hover:bg-[#111827]/50">
                      <td className="py-2 px-3 text-slate-300">
                        {meta?.description || channel} ({channel})
                      </td>
                      <td className="py-2 px-3">
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold text-slate-200">
                            {beforeVal !== undefined ? Number(beforeVal).toFixed(2) : '--'}{' '}
                            <span className="text-[10px] text-slate-500">{meta?.units}</span>
                          </span>
                          {renderProvenanceBadge('MEASURED', 'xs')}
                        </div>
                      </td>
                      <td className="py-2 px-3">
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold text-amber-300">
                            {afterVal !== undefined ? Number(afterVal).toFixed(2) : '--'}{' '}
                            <span className="text-[10px] text-slate-500">{meta?.units}</span>
                          </span>
                          {renderProvenanceBadge('SIMULATED', 'xs')}
                        </div>
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`font-semibold ${
                            Math.abs(delta) > 0.001
                              ? delta > 0
                                ? 'text-amber-400'
                                : 'text-blue-400'
                              : 'text-slate-500'
                          }`}
                        >
                          {delta > 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-slate-400">
                        {simulationResult.update_applied ? 'INCORPORATED' : 'REVERTED'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Recovery Plan if triggered */}
          {simulationResult.recovery_plan && (
            <div className="p-3 bg-amber-950/20 border border-amber-700/50 rounded-lg text-xs font-mono space-y-1">
              <div className="flex items-center gap-2 text-amber-400 font-bold uppercase">
                <span>Safe Recovery Advisory Plan Generated:</span>
              </div>
              <p className="text-slate-300">
                Action: {simulationResult.recovery_plan.action || 'Revert candidate state perturbation to baseline telemetry'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Section 5: Academic Simulation Non-Flight Notice Box */}
      <div className="p-3.5 bg-[#0B0F19] rounded-lg border border-amber-900/50 flex items-start gap-3 text-xs">
        <span className="text-amber-400 text-base font-bold select-none">⚠</span>
        <div className="space-y-1">
          <div className="font-mono font-bold text-amber-400 uppercase tracking-wide">
            Digital Twin Academic Non-Flight Operations Notice
          </div>
          <p className="text-slate-400 leading-relaxed">
            All Digital Twin What-If scenario perturbations (thermal stress, sensor bias) are synthetic candidate states evaluated inside a simulated virtual sandbox.
            Simulated outputs are strictly isolated from recorded NASA C-MAPSS dataset baselines and must never be interpreted as certified hardware telemetry or applied to live aircraft control systems.
          </p>
        </div>
      </div>
    </div>
  );
};

export default DigitalTwinSimulationPanel;
