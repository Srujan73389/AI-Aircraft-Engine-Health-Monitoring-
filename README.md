# AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/dataset-NASA%20C--MAPSS-orange.svg)](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq)
[![License](https://img.shields.io/badge/license-Academic%20Research-green.svg)](#disclaimer)

---

> **Academic Disclaimer:** This repository is an academic and research engineering prototype developed for final-year engineering research. It utilizes simulated turbofan degradation data from the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark. It is designed for methodology evaluation and educational demonstration and is **not certified or deployed for real-world aircraft operations or live flight control**.

---

## 1. Project Overview

In commercial and defense aviation, gas turbine engines are high-capital, mission-critical assets. In-flight component failures, unscheduled engine removals, and uncontained mechanical breakdowns incur severe operational disruptions, financial penalties, and flight-safety hazards. 

This project develops an end-to-end intelligent predictive maintenance framework for high-bypass aircraft turbofan engines. By combining physics-informed domain understanding, advanced multivariate machine learning, digital twin simulation, and agentic AI reasoning, the system continuously tracks engine degradation, predicts Remaining Useful Life (RUL), diagnoses subsystem fault modes, and recommends actionable maintenance decisions before catastrophic failure occurs.

---

## 2. Problem Statement

Traditional aircraft maintenance primarily relies on fixed flight-hour intervals or reactive servicing after alarms occur. Fixed schedules often lead to either premature component replacement (incurring unnecessary operational costs) or delayed intervention (risking secondary damage and unplanned groundings). 

Modern aircraft engines generate rich streams of sensor telemetry across variable flight regimes. The primary engineering challenges are:
1. **Operating Condition Complexity:** Variations in altitude, Mach number, and throttle angle produce large operational swings that obscure underlying component degradation.
2. **Multi-Component Fault Dynamics:** Degradation may originate in the High-Pressure Compressor (HPC), Fan, or both simultaneously, each with distinct wear rates.
3. **Sensor Heterogeneity:** Telemetry channels range from invariant environmental measurements to dynamic thermodynamic parameters requiring robust feature selection and regime-aware normalization.
4. **Prognostics and Health Management (PHM):** Accurate RUL prediction must be combined with autonomous, safety-verified mitigation decisions.

---

## 3. Project Objectives

- **Phase 1 – Dataset Setup & Verification:** Ingest, verify cryptographic immutability (SHA-256), and document the NASA C-MAPSS dataset.
- **Phase 2 – Dataset Understanding & Exploratory Data Analysis:** Conduct programmatic data quality audits, trajectory tracking, and distribution comparisons across all four subsets (FD001–FD004).
- **Phase 3 – Preprocessing & Feature Engineering:** Perform regime clustering, operating-condition normalization, feature selection, and RUL target formulation.
- **Phase 4 – Remaining Useful Life (RUL) Modeling:** Train and evaluate statistical, machine-learning, and deep sequential models (e.g. CNNs, LSTMs, Transformers) to predict RUL.
- **Phase 5 – Digital Twin Simulation:** Construct a high-fidelity software twin to simulate thermodynamic degradation and test counterfactual operating scenarios.
- **Phase 6 – Agentic AI & Autonomous Decision-Making:** Deploy intelligent agent workflows that reason over health indicators and recommend maintenance strategies.
- **Phase 7 – Safety Verification & Fault Mitigation:** Implement guardrails and fail-safe verification to ensure generated recommendations respect flight safety limits.
- **Phase 8 – Full-Stack Telemetry Dashboard:** Deliver an interactive web platform for real-time fleet health monitoring, RUL forecasting, and maintenance alerts.
- **Phase 9 – System Optimization & Deployment:** Benchmark inference latencies, package containers, and prepare final research documentation.

---

## 4. Project Architecture

The overall multi-phase system architecture integrates data ingestion, preprocessing, health assessment, RUL estimation, digital twin tracking, and autonomous agentic decisions:

![Project Architecture](docs/architecture.png.jpg)

---

## 5. Dataset Information: NASA C-MAPSS

The project benchmarks against the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset, created by the NASA Prognostics Center of Excellence (PCoE) at NASA Ames Research Center.

The benchmark models a commercial 90,000 lbf thrust class turbofan engine and is organized into four subsets reflecting escalating operational and fault complexities:

| Subset | Operating Conditions | Fault Modes | Training Trajectories (Engines) | Testing Trajectories (Engines) | Ground-Truth RULs | Total Operational Rows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FD001** | 1 (Single Condition) | 1 (HPC Degradation) | 100 engines (20,631 rows) | 100 engines (13,096 rows) | 100 values | 33,727 |
| **FD002** | 6 (Multi-Condition) | 1 (HPC Degradation) | 260 engines (53,759 rows) | 259 engines (33,991 rows) | 259 values | 87,750 |
| **FD003** | 1 (Single Condition) | 2 (HPC + Fan) | 100 engines (24,720 rows) | 100 engines (16,596 rows) | 100 values | 41,316 |
| **FD004** | 6 (Multi-Condition) | 2 (HPC + Fan) | 249 engines (61,249 rows) | 248 engines (41,214 rows) | 248 values | 102,463 |
| **Total** | — | — | **709 engines (160,359 rows)**| **707 engines (104,897 rows)**| **707 values** | **265,256 rows** |

### Telemetry Variables (26 Columns)
Each operational record represents one snapshot taken during a flight cycle:
- **Identifier:** `unit_number` (Engine ID, positive integer).
- **Time Index:** `time_cycles` (Discrete flight cycle counter, starting at 1).
- **Operational Settings:**
  - `op_setting_1`: Flight Altitude (0 to 42,000 ft).
  - `op_setting_2`: Mach Number (0.00 to 0.84).
  - `op_setting_3`: Throttle Resolver Angle (TRA: 60 to 100).
- **Sensor Measurements (21 continuous channels):**
  - Temperatures: Fan inlet ($T_2$), LPC outlet ($T_{24}$), HPC outlet ($T_{30}$), LPT outlet ($T_{50}$).
  - Pressures: Fan inlet ($P_2$), Bypass duct ($P_{15}$), HPC outlet total ($P_{30}$), HPC outlet static ($Ps_{30}$).
  - Spool Speeds: Physical fan speed ($N_f$), Physical core speed ($N_c$), Corrected fan speed ($Nf_{\text{dmd}}$), Corrected core speed ($Nc_{\text{dmd}}$).
  - Ratios & Mass Flows: Engine Pressure Ratio ($\text{epr}$), Fuel flow to $Ps_{30}$ ratio ($\phi$), Bypass Ratio ($\text{BPR}$), Burner Fuel-Air ratio ($\text{farB}$), Bleed enthalpy ($ht\text{Bleed}$), Demanded speeds ($Nf_{\text{dmd2}}$, $PCNfR_{\text{dmd}}$), Coolant bleeds ($W_{31}$, $W_{32}$).

For complete documentation on the dataset source, citations, and hash records, refer to [`docs/dataset_source.md`](docs/dataset_source.md).

---

## 6. Completed Work: Phase 1 & Phase 2

### Phase 1: Dataset Setup and Verification
- Ingested all 12 primary files and supporting documentation from the NASA repository.
- Established and verified dataset immutability using SHA-256 cryptographic hashes.
- Audited file headers, line terminators, and byte counts.
- Created comprehensive dataset documentation in [`docs/dataset_source.md`](docs/dataset_source.md).

### Phase 2: Dataset Understanding and Exploratory Data Analysis (EDA)
- **Step 2.1 – Inspection:** Confirmed 26-column tabular layout and initial record distributions.
- **Step 2.2 – Column Mapping:** Mapped columns to physical turbofan components per the PHM 2008 literature.
- **Step 2.3 – Boundary Check:** Analyzed first and last cycle records for all 12 dataset files.
- **Step 2.4 – Statistical Summary:** Computed total rows, columns, engine counts, and cycle lengths across subsets.
- **Step 2.5 – Data Quality Audit:** Verified zero missing values, zero non-numeric tokens, zero infinite values, and valid positive integer identifiers across all 265,256 rows.
- **Step 2.6 – Duplicate Analysis:** Confirmed zero full-row duplicates and zero key collisions on `(unit_number, time_cycles)`.
- **Step 2.7 – Chronological Continuity:** Programmatically verified that cycles increment strictly by $+1$ without gaps or reversals, and engine records form contiguous blocks.
- **Step 2.8 – Summary Statistics:** Computed parametric summary statistics (mean, standard deviation, quartiles, coefficient of variation) for all 26 columns across all 8 files.
- **Step 2.9 – Sensor Trajectories:** Generated 44 high-resolution trend plots across 11 key sensors for representative short, medium, and long engines.
- **Step 2.10 – Comparative Distributions:** Generated 42 comparative boxplots comparing training and testing distributions side-by-side.
- **Step 2.11 – Sensor Variation Classification:** Evaluated all 21 sensors across all four subsets into Constant, Near-constant, Low-variation, and Variable categories.
- **Step 2.12 – EDA Reporting:** Compiled the master [CMAPSS EDA Report](reports/eda/CMAPSS_EDA_Report.md) and [EDA Summary](reports/eda/EDA_Summary.md).

---

## 7. Key EDA Results & Important Findings

1. **Pristine Data Quality:** Zero missing, corrupted, duplicate, or disordered records exist across all 265,256 observations.
2. **Single vs. Multiple Condition Dynamics:**
   - In single-condition subsets (**FD001** and **FD003**), operating settings remain constant. Exactly 5 to 6 sensors (`sensor_1`, `sensor_5`, `sensor_16`, `sensor_18`, `sensor_19`, and `sensor_10` in FD001) exhibit **zero within-subset variance**. Active thermodynamic sensors exhibit progressive intra-engine drift toward end-of-life.
   - In multi-condition subsets (**FD002** and **FD004**), operation across six flight regimes causes all 21 sensors to exhibit non-zero variance. Operating regime shifts introduce wide multi-modal banding that obscures degradation in unnormalized plots.
3. **Consistently Low-Variation Sensors:** `sensor_15` (Bypass Ratio) maintains a standard deviation below $1.0$ across all four subsets ($0.04$ in FD001, $0.75$ in FD002, $0.06$ in FD003, and $0.75$ in FD004).
4. **Reproducible Train/Test Consistency:** Independent evaluation confirmed **100% classification agreement (84/84)** between training and testing sets.
5. **Phase 3 Preprocessing Implications:**
   - Invariant sensors in FD001/FD003 are primary candidates for feature pruning in linear/distance-based models.
   - In FD002/FD004, all sensors carry regime information and should not be globally dropped; regime clustering and condition normalization are essential candidates for experimentation.

---

## 8. Repository Folder Structure

```text
Aircraft_Engine_AI/
├── .gitignore                          # Excludes raw data, virtual environments, caches
├── README.md                           # Master project documentation
├── docs/
│   ├── architecture.png.jpg            # System architecture diagram
│   └── dataset_source.md               # NASA C-MAPSS provenance and verification record
├── dataset/
│   ├── raw/                            # NASA C-MAPSS raw files (excluded from Git)
│   └── processed/                      # Preprocessed data directory (preserved via .gitkeep)
│       └── .gitkeep
├── preprocessing/                      # Completed EDA & verification scripts
│   ├── inspect_dataset.py              # Step 2.1: Dataset structure inspection
│   ├── inspect_first_last_rows.py      # Step 2.3: Boundary cycle inspection
│   ├── summarize_subsets.py            # Step 2.4: Subset dimensions & cycle counts
│   ├── check_data_quality.py           # Step 2.5: Missing & non-numeric audit
│   ├── check_duplicates.py             # Step 2.6: Full-row & key duplicate check
│   ├── check_cycle_order.py            # Step 2.7: Chronological continuity audit
│   ├── summary_statistics.py           # Step 2.8: Parametric summary statistics
│   ├── plot_sensor_trends.py           # Step 2.9: Sensor trajectory visualization
│   ├── compare_sensor_distributions.py # Step 2.10: Comparative boxplot generation
│   └── classify_sensor_variation.py    # Step 2.11: Sensor variation classification
└── reports/
    └── eda/                            # Master EDA reports and outputs
        ├── CMAPSS_EDA_Report.md        # Comprehensive technical EDA report
        ├── EDA_Summary.md              # Executive EDA summary
        ├── train_FD001_summary.csv     # FD001 training statistics
        ├── test_FD001_summary.csv      # FD001 testing statistics
        ├── train_FD002_summary.csv     # FD002 training statistics
        ├── test_FD002_summary.csv      # FD002 testing statistics
        ├── train_FD003_summary.csv     # FD003 training statistics
        ├── test_FD003_summary.csv      # FD003 testing statistics
        ├── train_FD004_summary.csv     # FD004 training statistics
        ├── test_FD004_summary.csv      # FD004 testing statistics
        ├── sensor_trends/              # 44 individual sensor trajectory PNG plots
        └── sensor_distributions/       # 42 distribution boxplots & classification CSVs
            ├── sensor_variation_classification.csv
            ├── sensor_variation_analysis.md
            ├── train_sensor_distribution_comparison.csv
            └── test_sensor_distribution_comparison.csv
```

---

## 9. Technologies Used

- **Programming Language:** Python 3.12+
- **Package & Environment Management:** `uv`
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Visualization:** `matplotlib`
- **Version Control:** Git & GitHub

---

## 10. Getting Started

### Prerequisites
- Python 3.12 or newer installed on your system.
- `uv` (recommended) or standard `pip` for dependency management.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Srujan73389/AI-Aircraft-Engine-Health-Monitoring-.git
   cd AI-Aircraft-Engine-Health-Monitoring-
   ```

2. **Create and Activate Virtual Environment:**
   Using `uv`:
   ```bash
   uv venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   uv pip install pandas numpy matplotlib
   ```

4. **Acquire the Dataset:**
   The raw dataset files are excluded from Git to comply with repository best practices. To replicate the analysis:
   - Download the C-MAPSS dataset from the [NASA Data Portal](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq) or [NASA PCoE Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
   - Extract the files into `dataset/raw/C-MAPSS/` so that files such as `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` reside in that directory.
   - Refer to [`docs/dataset_source.md`](docs/dataset_source.md) for full provenance and SHA-256 verification hashes.

5. **Run the EDA Pipeline:**
   Execute any of the verification scripts directly from the workspace root:
   ```bash
   # Check data quality across all 8 files
   python preprocessing/check_data_quality.py

   # Check chronological cycle continuity
   python preprocessing/check_cycle_order.py

   # Generate parametric statistics
   python preprocessing/summary_statistics.py

   # Classify sensor variation
   python preprocessing/classify_sensor_variation.py
   ```

---

## 11. Current Status and Upcoming Roadmap

- [x] **Phase 1: Dataset Setup and Cryptographic Verification** *(Complete)*
- [x] **Phase 2: Dataset Understanding and Exploratory Data Analysis** *(Complete)*
- [ ] **Phase 3: Data Preprocessing & Feature Engineering** *(Next)*
- [ ] **Phase 4: RUL Prognostics & Failure Classification Models**
- [ ] **Phase 5: High-Fidelity Turbofan Digital Twin Simulation**
- [ ] **Phase 6: Multi-Agent AI System for Autonomous Decision Support**
- [ ] **Phase 7: Safety Verification & Guardrail Architecture**
- [ ] **Phase 8: Full-Stack Web Platform (FastAPI & React Telemetry Dashboard)**
- [ ] **Phase 9: Comprehensive Validation, Benchmarking & Deployment**

---

## 12. References & Citations

1. A. Saxena, K. Goebel, D. Simon, and N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, in Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, Oct 2008.
2. D. K. Frederick, J. A. DeCastro, and J. S. Litt, *"User's Guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)"*, NASA Technical Memorandum NASA/TM-2007-215026, 2007.
3. NASA Prognostics Center of Excellence (PCoE) Data Set Repository, NASA Ames Research Center.

---

## 13. License & Academic Use

This project is licensed for academic, research, and non-commercial educational use. All raw dataset rights belong to the National Aeronautics and Space Administration (NASA).
