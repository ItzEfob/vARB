# Autonomous Retirement Blackbox (ARB)
**Zero-Trust Failsafe Passivation & Predictive End-of-Life Architecture for LEO Assets**

[![Status](https://img.shields.io/badge/Status-Flight%20Ready-00ffcc.svg)]()
[![AI Architecture](https://img.shields.io/badge/AI-Multivariate%20Edge%20Inference-ffaa00.svg)]()
[![Hardware Loop](https://img.shields.io/badge/Failsafe-3--Min%20Debounce%20%7C%2024--Hr%20Coma-ff0044.svg)]()

---

## Executive Summary
Low Earth Orbit (LEO) is facing a catastrophic debris crisis. Current satellite retirement protocols rely entirely on active ground intervention and functioning radio links. If an asset suffers an unexpected power short, an explosive micrometeoroid impact, or a fatal CPU radiation fault, ground control loses communication and the asset becomes an uncontrollable, tumbling piece of space debris forever.

The **Autonomous Retirement Blackbox (ARB)** is a self-contained, galvanically isolated edge-computing failsafe. Acting as an immutable hardware dead-man's switch, the ARB continuously monitors the host asset's sensor bus via one-way optoisolators. Using a highly optimized, dual-brain machine learning architecture running locally in active RAM, the ARB autonomously detects critical trauma or predicts long-term resource starvation, actively hijacking the thruster relays to execute a standard **11 m/s apogee-raising graveyard maneuver** before the asset crosses the point of no return.

---

## The Dual-Brain Edge AI Architecture

Instead of relying on rigid, hardcoded thresholds that get easily confused by natural cosmic ray sensor glitches, the ARB employs probabilistic machine learning calibrated strictly on pre-mission nominal baselines.

1. **The Tactical Brain (Calibrated Isolation Forest):** Hunts for immediate multivariate structural breaks (tumbles, shorts, impacts). Standardizes heavy PSI readings alongside tiny spin rates (`StandardScaler`) and applies a strict **3-minute persistent debounce filter** to completely eliminate false alarms caused by natural baseline noise floor jitter.
2. **The Strategic Brain (Leakage-Free Linear Regression):** Protects against slow-moving resource starvation. By locking off random shuffling (`shuffle=False`) to prevent time-series data leakage, it reverse-engineers the physical valve depletion rates to predict the exact future epoch the propellant tank will breach the **15.0 PSI** limit required to execute the disposal burn.

---

## Repository Navigation for Judges

To evaluate the software architecture quickly, please follow this directory mapping:

    ARB_Hackathon_Project/
    │
    ├── Data/
    │   └── arb_telemetry_dataset.csv         # Generated Digital Twin datastream
    │
    ├── Notebooks/                            # Visual Proof & Model Training
    │   ├── tactical_brain.ipynb              # Debounced anomaly detection graphs
    │   └── strategic_brain.ipynb             # Long-term 15 PSI depletion forecasts
    │
    ├── Simulation/                           # NASA GMAT Verification Mechanics
    │   └── arb_graveyard_passivation.script  # The orbital mechanics proof
    │
    ├── src/                                  # Embedded Hardware Core OS
    │   ├── telemetry_generator.py            # High-Fidelity Physics & Trauma Engine
    │   ├── state_machine.py                  # Master OS Controller & Override Loop
    │   └── trained_models/                   # Frozen edge-inference files (.pkl)
    │       ├── sensor_scaler.pkl
    │       ├── tactical_brain.pkl
    │       └── strategic_brain.pkl
    │
    └── README.md                             # You are here

---

## Quick Start Guide (Evaluation Walkthrough)

Judges can execute the complete end-to-end mission lifecycle locally in 3 straightforward steps. Make sure you have the required dependencies installed:

    pip install pandas numpy scikit-learn matplotlib joblib

### Step 1: Generate the Digital Twin Datastream (vARB)
Because real orbital propulsion telemetry is highly proprietary, we built a custom **High-Fidelity Digital Twin Simulator**. It models an empirical station-keeping depletion curve injected with realistic Gaussian sensor noise ($\pm 0.5$ PSI) while randomly triggering complex multi-variable disasters mid-flight.

    python src/telemetry_generator.py

*Output: Overwrites `Data/arb_telemetry_dataset.csv` with a fresh 2,000-epoch timeline and outputs a terminal sneak peek of the exact failure injection point.*

### Step 2: Verify the AI Visual Proofs (Optional)
Open the Jupyter Notebooks inside the `Notebooks/` directory and run all cells to observe exactly how the models handle baseline calibration, debounce persistence, and deep-future extrapolation.

*Outputs: Generates presentation-ready dark-mode dashboards directly into the `Data/` directory (`isolation_forest_bulletproof.png` and `linear_regression_forecast.png`).*

### Step 3: Run the Embedded Master OS Controller (pARB Core Demo)
Execute the primary flight loop that lives inside the physical ARB storage chips. This script loads the frozen `.pkl` brain files in milliseconds, reads incoming edge telemetry tick-by-tick, enforces the **24-Hour Coma Protocol** (hardware heartbeat override), and actively drops the physical relay hammer when failsafe bounds are breached.

    python src/state_machine.py

*Output: Watch the flight computer cycle safely through the `NOMINAL` state, switch instantly to `ARMED` to debounce sensor noise, and violently output the `⚡ PHYSICAL OVERRIDE ENGAGED` execution banner exactly when the physical parameters break.*

---

## Physical Verification (NASA GMAT)
To prove the physical validity of our software override, we have included an orbital mechanics propagation script (`Simulation/arb_graveyard_passivation.script`) built for **NASA's General Mission Analysis Tool (GMAT)**.

Loading this script models our host CubeSat in a $700\text{ km}$ LEO shell. Triggering our failsafe executes an $11\text{ m/s}$ blow-down prograde burn, mathematically proving that the maneuver successfully raises the asset's apogee clear of congested operational traffic while completely venting residual propellant mass.

---

## Key Technical Innovations

* **Zero Data Leakage:** Models are trained strictly via nominal gatekeeping (`Target_Status == 'Nominal'`), completely blinding the AI to future disaster states to prove pure structural inference.
* **Sensor Uncertainty Respect (Debouncing):** A single anomalous reading never triggers disposal. The ARB strictly requires 3 consecutive epochs of verified AI trauma before hijacking the ship, eliminating premature mission termination.
* **The Coma Protocol:** Independent of the AI, the master hardware loop tracks host CPU ping times. If the host suffers total radio/CPU death (`999ms` pings) for 24 continuous hours, the ARB bypasses all intelligence and physically discharges its supercapacitors to force passivation.
* **Decoupled Power Topology:** Splits processing into a highly lightweight, active edge reflex (Tactical Forest) and a highly specialized, periodic sleep-mode planner (Strategic Regression) to preserve strict CubeSat power budgets.

---
*Built with absolute rigor for a cleaner, safer Low Earth Orbit.*
