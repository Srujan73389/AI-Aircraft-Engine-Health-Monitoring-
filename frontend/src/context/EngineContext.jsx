import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import apiService from '../services/api';

const EngineContext = createContext(null);

export const EngineProvider = ({ children }) => {
  const [selectedSubset, setSelectedSubset] = useState('FD001');
  const [selectedUnit, setSelectedUnit] = useState(1);
  const [selectedCycle, setSelectedCycle] = useState(1);
  const [maxCycles, setMaxCycles] = useState(192);

  const [availableSubsets, setAvailableSubsets] = useState([]);
  const [availableEngines, setAvailableEngines] = useState([]);
  const [cycleState, setCycleState] = useState(null);
  const [sensorTrends, setSensorTrends] = useState(null);
  const [trendsLoading, setTrendsLoading] = useState(false);
  const [anomalyData, setAnomalyData] = useState(null);
  const [anomalyLoading, setAnomalyLoading] = useState(false);
  const [prognosticsData, setPrognosticsData] = useState(null);
  const [prognosticsLoading, setPrognosticsLoading] = useState(false);
  const [simulationResult, setSimulationResult] = useState(null);
  const [simulationLoading, setSimulationLoading] = useState(false);
  const [auditRecords, setAuditRecords] = useState([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [validationError, setValidationError] = useState(null);

  const clearValidationError = useCallback(() => {
    setValidationError(null);
  }, []);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendOnline, setBackendOnline] = useState(true);

  const [isPlaying, setIsPlaying] = useState(false);
  const playTimerRef = useRef(null);

  // 1. Initial health ping & subsets loading
  const fetchAuditRecords = useCallback(async (decisionStatus = null) => {
    setAuditLoading(true);
    try {
      const params = {};
      if (decisionStatus && decisionStatus !== 'ALL') {
        params.decision_status = decisionStatus.toUpperCase();
      }
      const data = await apiService.getAuditTrail(params);
      setAuditRecords(data?.records || []);
    } catch (err) {
      console.warn('Failed to load safety audit trail:', err);
    } finally {
      setAuditLoading(false);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      try {
        await apiService.checkHealth();
        if (isMounted) setBackendOnline(true);
      } catch (err) {
        if (isMounted) setBackendOnline(false);
      }

      try {
        const subsets = await apiService.getFleetSubsets();
        if (isMounted) {
          setAvailableSubsets(subsets);
        }
      } catch (err) {
        console.warn('Failed to load fleet subsets:', err);
      }

      // Load initial safety audit records
      fetchAuditRecords();
    };
    init();
    return () => { isMounted = false; };
  }, [fetchAuditRecords]);

  // 2. Load engines when subset changes
  useEffect(() => {
    let isMounted = true;
    const fetchEngines = async () => {
      try {
        const engines = await apiService.getSubsetEngines(selectedSubset);
        if (isMounted) {
          setAvailableEngines(engines);
          if (engines.length > 0 && !engines.includes(selectedUnit)) {
            setSelectedUnit(engines[0]);
          }
        }
      } catch (err) {
        console.warn(`Failed to fetch engines for ${selectedSubset}:`, err);
      }
    };
    fetchEngines();
    return () => { isMounted = false; };
  }, [selectedSubset]);

  // 3. Load trajectory sensor trends and dynamically set maxCycles
  useEffect(() => {
    let isMounted = true;
    const fetchTrends = async () => {
      setTrendsLoading(true);
      try {
        const sensors = await apiService.getEngineSensors(selectedSubset, selectedUnit);
        if (isMounted && sensors) {
          setSensorTrends(sensors);
          if (sensors.total_cycles) {
            setMaxCycles(sensors.total_cycles);
            if (selectedCycle > sensors.total_cycles) {
              setSelectedCycle(sensors.total_cycles);
            }
          }
        }
      } catch (err) {
        console.warn('Failed to resolve trajectory sensor trends:', err);
      } finally {
        if (isMounted) setTrendsLoading(false);
      }
    };
    fetchTrends();
    return () => { isMounted = false; };
  }, [selectedSubset, selectedUnit]);

  // 4. Load cycle state vector from backend
  const fetchCycleState = useCallback(async (subset, unit, cycle) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getEngineCycleState(subset, unit, cycle);
      setCycleState(data);
      setBackendOnline(true);
      try {
        setAnomalyLoading(true);
        // Extract clean numeric sensor readings map from state vector
        const rawState = data?.state_vector || {};
        const sensorMap = {};
        for (const [k, v] of Object.entries(rawState)) {
          if (typeof v === 'number' && !isNaN(v)) {
            sensorMap[k] = v;
          }
        }

        const anom = await apiService.checkAnomaly({
          subset,
          unit_number: unit,
          cycle,
          sensor_readings: sensorMap,
        });
        setAnomalyData(anom);
      } catch (anomErr) {
        console.warn('Anomaly evaluation fallback:', anomErr);
      } finally {
        setAnomalyLoading(false);
      }

      // Fetch prognostics RUL predictions
      try {
        setPrognosticsLoading(true);
        const prog = await apiService.predictRUL({
          subset,
          unit_number: unit,
          cycle,
        });
        setPrognosticsData(prog);
      } catch (progErr) {
        console.warn('Prognostics prediction fallback:', progErr);
      } finally {
        setPrognosticsLoading(false);
      }
    } catch (err) {
      console.error(`Failed to fetch cycle state for ${subset} U${unit} C${cycle}:`, err);
      const detail = err.response?.data?.detail || err.message || 'Failed to fetch engine state';
      setError(detail);

      const status = err.response?.status;
      if (status === 404) {
        setValidationError({
          category: 'INVALID_ENGINE_ID',
          severity: 'ERROR',
          title: `Engine ID Out of Range: Unit ${unit}`,
          message: `The requested engine unit ${unit} does not exist in dataset subset ${subset}.`,
          detail,
          suggestion: 'Select an engine ID within the valid operational range for this fleet subset.',
        });
      } else if (status === 400 && String(detail).toLowerCase().includes('cycle')) {
        setValidationError({
          category: 'INVALID_ENGINE_ID',
          severity: 'WARNING',
          title: `Flight Cycle Exceeded: Cycle ${cycle}`,
          message: `The requested flight cycle ${cycle} exceeds maximum trajectory history for Engine ${unit}.`,
          detail,
          suggestion: 'Adjust cycle index using scrubber or stepper to stay within recorded cycle boundaries.',
        });
      } else if (status === 422) {
        setValidationError({
          category: 'MISSING_INPUT',
          severity: 'ERROR',
          title: 'Missing or Malformed Request Parameters',
          message: 'The backend rejected the telemetry request due to missing or invalid fields.',
          detail,
          suggestion: 'Verify all required parameters are provided in valid format.',
        });
      } else if (status === 503 || String(detail).includes('MODEL_UNAVAILABLE')) {
        setValidationError({
          category: 'MODEL_UNAVAILABLE',
          severity: 'ERROR',
          title: 'Prognostic Model / Scaler Unavailable',
          message: 'Required champion model checkpoint or scaling pipeline file is unavailable on server.',
          detail,
          suggestion: 'Ensure Phase 3 scalers and Phase 4 model checkpoints are installed and accessible.',
        });
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCycleState(selectedSubset, selectedUnit, selectedCycle);
  }, [selectedSubset, selectedUnit, selectedCycle, fetchCycleState]);

  // 5. Subset selection handler
  const handleSelectSubset = useCallback((subset) => {
    if (subset !== selectedSubset) {
      clearValidationError();
      setSelectedSubset(subset);
      setSelectedUnit(1);
      setSelectedCycle(1);
      setIsPlaying(false);
    }
  }, [selectedSubset, clearValidationError]);

  // 6. Unit selection handler with client-side defensive validation
  const handleSelectUnit = useCallback((unit) => {
    const unitInt = parseInt(unit, 10);
    if (isNaN(unitInt) || unitInt < 1) {
      setValidationError({
        category: 'INVALID_ENGINE_ID',
        severity: 'ERROR',
        title: 'Invalid Engine Identifier',
        message: `Engine ID '${unit}' is invalid. Unit identifier must be a positive integer >= 1.`,
        suggestion: 'Enter a valid engine unit number from 1 to fleet maximum.',
      });
      return;
    }
    if (availableEngines.length > 0 && !availableEngines.includes(unitInt)) {
      setValidationError({
        category: 'INVALID_ENGINE_ID',
        severity: 'ERROR',
        title: `Engine ID Out of Range: Unit ${unitInt}`,
        message: `Engine Unit ${unitInt} not found in subset '${selectedSubset}'. Valid range is 1 to ${Math.max(...availableEngines)}.`,
        suggestion: `Select an engine ID between 1 and ${Math.max(...availableEngines)}.`,
      });
      return;
    }
    clearValidationError();
    if (unitInt !== selectedUnit) {
      setSelectedUnit(unitInt);
      setSelectedCycle(1);
      setIsPlaying(false);
    }
  }, [selectedUnit, availableEngines, selectedSubset, clearValidationError]);

  // 7. Cycle selection & stepper handlers with bounds validation
  const handleSelectCycle = useCallback((cycle) => {
    const cycleInt = parseInt(cycle, 10);
    if (isNaN(cycleInt) || cycleInt < 1) {
      setValidationError({
        category: 'INVALID_ENGINE_ID',
        severity: 'ERROR',
        title: 'Invalid Cycle Index',
        message: `Cycle index '${cycle}' is invalid. Operational cycle must be >= 1.`,
        suggestion: 'Specify a valid flight cycle >= 1.',
      });
      return;
    }
    if (maxCycles > 0 && cycleInt > maxCycles) {
      setValidationError({
        category: 'INVALID_ENGINE_ID',
        severity: 'WARNING',
        title: `Cycle Exceedance: Cycle ${cycleInt}`,
        message: `Requested cycle ${cycleInt} exceeds maximum recorded trajectory of ${maxCycles} cycles for Engine ${selectedUnit}.`,
        suggestion: `Select a cycle between 1 and ${maxCycles}.`,
      });
      return;
    }
    clearValidationError();
    const clamped = Math.max(1, Math.min(cycleInt, maxCycles));
    setSelectedCycle(clamped);
  }, [maxCycles, selectedUnit, clearValidationError]);

  const stepCycle = useCallback((delta) => {
    clearValidationError();
    setSelectedCycle((prev) => {
      const target = prev + delta;
      return Math.max(1, Math.min(target, maxCycles));
    });
  }, [maxCycles, clearValidationError]);

  // 8. Replay Play/Pause controller
  const togglePlayReplay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // 9. Closed-loop What-If Simulation Runner
  const runSimulation = useCallback(async (scenarioPayload) => {
    setSimulationLoading(true);
    try {
      const res = await apiService.runSimulationWhatIf(scenarioPayload);
      setSimulationResult(res);
      clearValidationError();
      // Refresh audit trail following simulation commit
      fetchAuditRecords();
      return res;
    } catch (err) {
      console.error('Simulation execution failed:', err);
      const status = err.response?.status;
      const detail = err.response?.data?.detail || err.message;
      if (status === 422) {
        setValidationError({
          category: 'MISSING_INPUT',
          severity: 'ERROR',
          title: 'What-If Simulation Input Rejected',
          message: 'The simulation request failed parameter validation (HTTP 422).',
          detail,
          suggestion: 'Ensure action type, target variable, and perturbation value are correctly specified.',
        });
      } else if (status === 503 || String(detail).includes('MODEL_UNAVAILABLE')) {
        setValidationError({
          category: 'MODEL_UNAVAILABLE',
          severity: 'ERROR',
          title: 'Simulation Model Checkpoint Missing',
          message: 'The Digital Twin simulator could not access required ML model inference checkpoints.',
          detail,
          suggestion: 'Verify champion models are loaded in backend model registry.',
        });
      }
      throw err;
    } finally {
      setSimulationLoading(false);
    }
  }, [fetchAuditRecords, clearValidationError]);

  const clearSimulation = useCallback(() => {
    setSimulationResult(null);
  }, []);

  useEffect(() => {
    if (isPlaying) {
      playTimerRef.current = setInterval(() => {
        setSelectedCycle((prev) => {
          if (prev >= maxCycles) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 350);
    } else {
      if (playTimerRef.current) {
        clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    }
    return () => {
      if (playTimerRef.current) {
        clearInterval(playTimerRef.current);
      }
    };
  }, [isPlaying, maxCycles]);

  const value = {
    selectedSubset,
    selectedUnit,
    selectedCycle,
    maxCycles,
    availableSubsets,
    availableEngines,
    cycleState,
    sensorTrends,
    trendsLoading,
    anomalyData,
    anomalyLoading,
    prognosticsData,
    prognosticsLoading,
    simulationResult,
    simulationLoading,
    runSimulation,
    clearSimulation,
    auditRecords,
    auditLoading,
    fetchAuditRecords,
    validationError,
    setValidationError,
    clearValidationError,
    loading,
    error,
    backendOnline,
    isPlaying,
    setSubset: handleSelectSubset,
    setUnit: handleSelectUnit,
    setCycle: handleSelectCycle,
    stepCycle,
    togglePlayReplay,
    refetchState: () => fetchCycleState(selectedSubset, selectedUnit, selectedCycle),
  };

  return (
    <EngineContext.Provider value={value}>
      {children}
    </EngineContext.Provider>
  );
};

export const useEngine = () => {
  const context = useContext(EngineContext);
  if (!context) {
    throw new Error('useEngine must be used within an EngineProvider');
  }
  return context;
};

export default EngineContext;
