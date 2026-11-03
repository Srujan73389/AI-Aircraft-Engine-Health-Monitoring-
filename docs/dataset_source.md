# NASA C-MAPSS Dataset Source & Documentation Record

**Project:** Aircraft Engine AI — Health Monitoring, Fault Diagnosis, and Predictive Maintenance  
**Roadmap Reference:** Phase 1, Step 1.6  
**Document Creation Date:** 3 November 2026  

---

## 1. Dataset Name
**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**  
(Commercial Modular Aero-Propulsion System Simulation)

---

## 2. Dataset Source & Reference
* **Originating Organization:** NASA Prognostics Center of Excellence (PCoE), NASA Ames Research Center.
* **Public Repository URLs:**
  * NASA Open Data Portal: [https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq)
  * NASA PCoE Data Repository: [https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
* **Original Reference Publication:**
  * A. Saxena, K. Goebel, D. Simon, and N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, in Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, Oct 2008.
* **Software Simulation Guide:**
  * D. K. Frederick, J. A. DeCastro, and J. S. Litt, *"User's Guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)"*, NASA Technical Memorandum NASA/TM-2007-215026, 2007.
* **Version Note:** NASA does not issue formal semantic version numbers (e.g., v1.0) for this benchmark package; it is officially identified by its PHM 2008 benchmark designation and sub-dataset identifiers (`FD001` through `FD004`).

---

## 3. Download Date
* **Date Downloaded:** 3 November 2026

---

## 4. Dataset Subsets
The dataset is structured into four distinct sub-datasets reflecting different levels of operational and fault complexity:

| Subset | Operating Conditions | Fault Modes | Description |
| :--- | :--- | :--- | :--- |
| **FD001** | ONE (Sea Level) | ONE (HPC Degradation) | Baseline single-regime, single-failure dataset |
| **FD002** | SIX Flight Regimes | ONE (HPC Degradation) | Multi-condition dataset across altitude/Mach/TRA settings |
| **FD003** | ONE (Sea Level) | TWO (HPC + Fan Degradation) | Dual fault progression under constant operating regime |
| **FD004** | SIX Flight Regimes | TWO (HPC + Fan Degradation) | Multi-condition, multi-fault benchmark dataset |

---

## 5. Dataset Contents & File Inventory
The dataset package consists of 14 files stored in the repository:

### Training Files (Run-to-failure operational cycles)
* `train_FD001.txt` — 20,631 records across 100 engines
* `train_FD002.txt` — 53,759 records across 260 engines
* `train_FD003.txt` — 24,720 records across 100 engines
* `train_FD004.txt` — 61,249 records across 249 engines

### Testing Files (Trajectories truncated prior to failure)
* `test_FD001.txt` — 13,096 records across 100 engines
* `test_FD002.txt` — 33,991 records across 259 engines
* `test_FD003.txt` — 16,596 records across 100 engines
* `test_FD004.txt` — 41,214 records across 248 engines

### Ground Truth RUL Files (Remaining Useful Life after cutoff)
* `RUL_FD001.txt` — 100 integer ground-truth values
* `RUL_FD002.txt` — 259 integer ground-truth values
* `RUL_FD003.txt` — 100 integer ground-truth values
* `RUL_FD004.txt` — 248 integer ground-truth values

### Original Documentation & Literature
* `readme.txt` — Original benchmark documentation text file
* `Damage Propagation Modeling.pdf` — Original PHM08 conference research paper

---

## 6. Dataset Format & Schema
* **File Type:** Space-separated numerical ASCII text files (`sep=r'\s+'`).
* **Trajectory Data Structure (`train_FD00x.txt`, `test_FD00x.txt`):** 26 space-separated numerical columns per row representing a single operational cycle snapshot:
  * **Col 1:** `unit_number` (Engine identifier integer: $1, 2, \dots, N$)
  * **Col 2:** `time_cycles` (Operational time cycle: $1, 2, \dots, T$)
  * **Col 3–5:** 3 Operational settings (`op_setting_1`: Altitude [kft], `op_setting_2`: Mach number, `op_setting_3`: Throttle Resolver Angle [TRA %])
  * **Col 6–26:** 21 Thermodynamic sensor measurements across engine stations ($T_2, T_{24}, T_{30}, T_{50}, P_2, P_{15}, P_{30}, N_f, N_c, epr, P_{s30}, \Phi, N_{Rf}, N_{Rc}, BPR, farB, htBleed, N_{f\_dmd}, PCNfR\_dmd, W_{31}, W_{32}$)
* **Ground Truth RUL Structure (`RUL_FD00x.txt`):** 1 integer per line corresponding to the remaining operational life after the final cycle of test engine unit $i$.

---

## 7. Dataset Location
* **Raw Files Directory:** `dataset/raw/C-MAPSS/`
* **Processed Data Directory:** `dataset/processed/`

---

## 8. Dataset Limitations
1. **Simulation Nature:** C-MAPSS data was synthesized using NASA's FORTRAN-based turbofan engine simulation model coupled with exponential damage propagation equations. It is **not live telemetry** recorded from commercial in-flight aircraft.
2. **Noise Characterization:** Sensor measurement contamination follows synthetic Gaussian noise distributions rather than complex physical environmental interference.
3. **Non-Authoritative Boundary:** Any predictive models, digital twin simulations, or recovery recommendations derived from this dataset are for **academic and engineering prototyping only**, and must never be interpreted as certified aircraft flight-control or airworthiness maintenance authorities.

---

## 9. Verification Reference
A comprehensive dataset inspection and verification was executed prior to pipeline development:
* **Integrity Validation:** 0 missing/null values, 0 duplicate rows, and 0 sequence gaps across all 12 dataset files.
* **Cycle Ordering:** Confirmed strictly monotonic incrementing cycles starting at cycle 1 for all engines.
* **Documentation Discrepancy Note:** The original `readme.txt` stated that FD004 contained 248 train trajectories and 249 test trajectories. Direct programmatic file inspection confirmed that `train_FD004.txt` has **249 engines** and `test_FD004.txt` has **248 engines**, perfectly aligning with `RUL_FD004.txt` (248 ground-truth entries). The readme reversed these two counts.
* **Immutability:** SHA-256 cryptographic hashes and modification timestamps confirm the raw dataset files remain unchanged.

---

## 10. Data Governance & Usage Notes
1. **Raw Data Immutability:** Files in `dataset/raw/C-MAPSS/` are read-only and must remain unmodified.
2. **Processed Data Separation:** All cleaned, scaled, feature-engineered datasets, labels, and intermediate arrays must be saved under `dataset/processed/`.
3. **Data Leakage Prevention:** Normalization parameters, scalers, and sequence configurations must be fitted exclusively on training trajectories and subsequently applied to validation and test subsets.
