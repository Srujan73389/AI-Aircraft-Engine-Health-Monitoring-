/**
 * NASA C-MAPSS Sensor Telemetry Metadata & Engineering Units Dictionary
 * Defines station locations, thermodynamic descriptions, and physical units
 * for all 21 raw sensor channels in accordance with NASA TM-2008-215106.
 *
 * Roadmap Reference: Phase 8 - Step 8.5
 */

export const SENSOR_METADATA = {
  sensor_1: {
    id: "sensor_1",
    name: "T2",
    station: "Station 2 - Fan Inlet",
    description: "Total temperature at fan inlet",
    units: "°R",
    category: "temperatures",
    provenance: "MEASURED",
    color: "#38BDF8", // Sky blue
  },
  sensor_2: {
    id: "sensor_2",
    name: "T24",
    station: "Station 2.4 - LPC Outlet",
    description: "Total temperature at LPC outlet",
    units: "°R",
    category: "temperatures",
    provenance: "MEASURED",
    color: "#60A5FA", // Blue
  },
  sensor_3: {
    id: "sensor_3",
    name: "T30",
    station: "Station 3 - HPC Outlet",
    description: "Total temperature at HPC outlet",
    units: "°R",
    category: "temperatures",
    provenance: "MEASURED",
    color: "#F59E0B", // Amber / thermal warning
  },
  sensor_4: {
    id: "sensor_4",
    name: "T50",
    station: "Station 5 - LPT Outlet",
    description: "Total temperature at LPT outlet (EGT)",
    units: "°R",
    category: "temperatures",
    provenance: "MEASURED",
    color: "#EF4444", // Red / EGT
  },
  sensor_5: {
    id: "sensor_5",
    name: "P2",
    station: "Station 2 - Fan Inlet",
    description: "Total pressure at fan inlet",
    units: "psia",
    category: "pressures",
    provenance: "MEASURED",
    color: "#06B6D4", // Cyan
  },
  sensor_6: {
    id: "sensor_6",
    name: "P15",
    station: "Station 1.5 - Bypass Duct",
    description: "Total pressure in fan bypass duct",
    units: "psia",
    category: "pressures",
    provenance: "MEASURED",
    color: "#0EA5E9", // Ocean
  },
  sensor_7: {
    id: "sensor_7",
    name: "P30",
    station: "Station 3 - HPC Outlet",
    description: "Total pressure at HPC outlet (Combustor Casing)",
    units: "psia",
    category: "pressures",
    provenance: "MEASURED",
    color: "#14B8A6", // Teal
  },
  sensor_8: {
    id: "sensor_8",
    name: "Nf",
    station: "Low-Pressure Spool",
    description: "Physical fan rotational speed",
    units: "rpm",
    category: "speeds",
    provenance: "MEASURED",
    color: "#10B981", // Emerald
  },
  sensor_9: {
    id: "sensor_9",
    name: "Nc",
    station: "High-Pressure Core Spool",
    description: "Physical core rotational speed",
    units: "rpm",
    category: "speeds",
    provenance: "MEASURED",
    color: "#8B5CF6", // Purple
  },
  sensor_10: {
    id: "sensor_10",
    name: "epr",
    station: "Exhaust Diffuser",
    description: "Engine pressure ratio (P50/P2)",
    units: "--",
    category: "pressures",
    provenance: "MEASURED",
    color: "#A855F7", // Violet
  },
  sensor_11: {
    id: "sensor_11",
    name: "Ps30",
    station: "Station 3 - HPC Outlet",
    description: "Static pressure at HPC casing wall",
    units: "psia",
    category: "pressures",
    provenance: "MEASURED",
    color: "#2DD4BF", // Mint
  },
  sensor_12: {
    id: "sensor_12",
    name: "phi",
    station: "Combustor Fuel System",
    description: "Ratio of fuel flow to static pressure Ps30",
    units: "pps/psia",
    category: "bleed_flow",
    provenance: "MEASURED",
    color: "#F97316", // Orange
  },
  sensor_13: {
    id: "sensor_13",
    name: "NRf",
    station: "Low-Pressure Spool",
    description: "Corrected fan rotational speed",
    units: "rpm",
    category: "speeds",
    provenance: "MEASURED",
    color: "#34D399", // Light emerald
  },
  sensor_14: {
    id: "sensor_14",
    name: "NRc",
    station: "High-Pressure Core Spool",
    description: "Corrected core rotational speed",
    units: "rpm",
    category: "speeds",
    provenance: "MEASURED",
    color: "#C084FC", // Light purple
  },
  sensor_15: {
    id: "sensor_15",
    name: "BPR",
    station: "Bypass Duct / Core Splitter",
    description: "Bypass Ratio (Fan bypass to core flow)",
    units: "--",
    category: "performance",
    provenance: "MEASURED",
    color: "#EC4899", // Pink
  },
  sensor_16: {
    id: "sensor_16",
    name: "farB",
    station: "Combustor Primary Zone",
    description: "Burner fuel-air mass ratio",
    units: "--",
    category: "performance",
    provenance: "MEASURED",
    color: "#FB923C", // Light orange
  },
  sensor_17: {
    id: "sensor_17",
    name: "htBleed",
    station: "Bleed Enthalpy Duct",
    description: "High-pressure customer bleed enthalpy",
    units: "lbf",
    category: "bleed_flow",
    provenance: "MEASURED",
    color: "#EAB308", // Yellow
  },
  sensor_18: {
    id: "sensor_18",
    name: "Nf_dmd",
    station: "FADEC Control Loop",
    description: "Demanded physical fan speed",
    units: "rpm",
    category: "speeds",
    provenance: "MEASURED",
    color: "#94A3B8", // Slate
  },
  sensor_19: {
    id: "sensor_19",
    name: "PCNfR_dmd",
    station: "FADEC Control Loop",
    description: "Demanded corrected fan speed",
    units: "%",
    category: "speeds",
    provenance: "MEASURED",
    color: "#64748B", // Slate dark
  },
  sensor_20: {
    id: "sensor_20",
    name: "W31",
    station: "HPT Stator Cooling",
    description: "High-Pressure Turbine coolant bleed flow",
    units: "lbm/s",
    category: "bleed_flow",
    provenance: "MEASURED",
    color: "#22D3EE", // Sky
  },
  sensor_21: {
    id: "sensor_21",
    name: "W32",
    station: "LPT Stator Cooling",
    description: "Low-Pressure Turbine coolant bleed flow",
    units: "lbm/s",
    category: "bleed_flow",
    provenance: "MEASURED",
    color: "#818CF8", // Indigo
  },
};

/**
 * Returns metadata definition by either sensor key ("sensor_3") or canonical name ("T30").
 */
export const getSensorMetadata = (key) => {
  if (!key) return null;
  if (SENSOR_METADATA[key]) {
    return SENSOR_METADATA[key];
  }
  const clean = String(key).toLowerCase().replace(/^g\d+_/, "");
  for (const meta of Object.values(SENSOR_METADATA)) {
    if (meta.name.toLowerCase() === clean) {
      return meta;
    }
  }
  return null;
};

/**
 * Formats a raw sensor numerical reading with its physical engineering units.
 */
export const formatSensorValue = (sensorKey, value) => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return "N/A";
  }
  const num = Number(value);
  const meta = getSensorMetadata(sensorKey);
  const unit = meta?.units || "";

  let formattedNum = "";
  if (unit === "rpm") {
    formattedNum = num.toFixed(1);
  } else if (unit === "°R" || unit === "psia") {
    formattedNum = num.toFixed(2);
  } else if (unit === "lbm/s" || unit === "pps/psia") {
    formattedNum = num.toFixed(3);
  } else if (unit === "%") {
    formattedNum = num.toFixed(1);
  } else if (unit === "--" || unit === "") {
    formattedNum = num.toFixed(4);
  } else {
    formattedNum = num.toFixed(2);
  }

  if (unit === "--" || unit === "") {
    return formattedNum;
  }
  if (unit === "%") {
    return `${formattedNum}%`;
  }
  return `${formattedNum} ${unit}`;
};

export default SENSOR_METADATA;
