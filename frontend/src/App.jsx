import React from 'react';
import { EngineProvider } from './context/EngineContext';
import Header from './components/layout/Header';
import EngineSelectorBar from './components/dashboard/EngineSelectorBar';
import EngineSummaryCard from './components/dashboard/EngineSummaryCard';
import SensorTrendsChart from './components/dashboard/SensorTrendsChart';
import SensorReadingsGrid from './components/dashboard/SensorReadingsGrid';
import AnomalyDiagnosisPanel from './components/dashboard/AnomalyDiagnosisPanel';
import PrognosticsRULPanel from './components/dashboard/PrognosticsRULPanel';
import DigitalTwinSimulationPanel from './components/dashboard/DigitalTwinSimulationPanel';
import AgentSafetyGovernancePanel from './components/dashboard/AgentSafetyGovernancePanel';
import ValidationAlertBanner from './components/common/ValidationAlertBanner';

export const App = () => {
  return (
    <EngineProvider>
      <div className="min-h-screen bg-[#0B0F19] text-[#F3F4F6] flex flex-col font-sans">
        {/* Mission Control Top Navigation & Disclaimer */}
        <Header />

        {/* Main Dashboard Canvas */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-6 space-y-6">
          {/* Defensive Validation Alert Banner (Phase 8 Step 8.10) */}
          <ValidationAlertBanner />

          {/* Subsystem 1: Engine Selector & Trajectory Replay Scrubber */}
          <section aria-label="Engine Selector and Cycle Control">
            <EngineSelectorBar />
          </section>

          {/* Subsystem 2: Engine Summary & Gas-Path Telemetry Card */}
          <section aria-label="Engine Status and Gas-Path Telemetry">
            <EngineSummaryCard />
          </section>

          {/* Subsystem 3: Multi-Sensor Trajectory Trends Chart */}
          <section aria-label="Sensor Trends Chart">
            <SensorTrendsChart />
          </section>

          {/* Subsystem 4: Full 21-Sensor Telemetry Matrix */}
          <section aria-label="Full Sensor Readings Grid">
            <SensorReadingsGrid />
          </section>

          {/* Subsystem 5: Anomaly Flags & Diagnostic Evidence Panel */}
          <section aria-label="Statistical Anomaly and Diagnostic Evidence">
            <AnomalyDiagnosisPanel />
          </section>

          {/* Subsystem 6: Prognostics Remaining Useful Life (RUL) Panel */}
          <section aria-label="Remaining Useful Life Prognostics">
            <PrognosticsRULPanel />
          </section>

          {/* Subsystem 7: Digital Twin State & Scenario What-If Sandbox */}
          <section aria-label="Digital Twin State and Scenario Outputs">
            <DigitalTwinSimulationPanel />
          </section>

          {/* Subsystem 8: Multi-Agent AI Workflow Logs & Safety Verification Governance */}
          <section aria-label="Agent Execution Logs and Safety Verification">
            <AgentSafetyGovernancePanel />
          </section>

          {/* Phase 8 Step 8.10 Information Banner */}
          <footer className="text-center py-4 border-t border-[#1F2937]/50 text-xs font-mono text-slate-500">
            NASA C-MAPSS Digital Twin Architecture • Phase 8 Milestone 8.10 Defensive Validation &amp; Alert System
          </footer>
        </main>
      </div>
    </EngineProvider>
  );
};

export default App;
