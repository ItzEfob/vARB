"""
ARB Telemetry Generator — High-Fidelity Physics & Trauma Engine

Simulates 2000 minutes of satellite sensor data, then injects one
of three catastrophic failure scenarios at a randomised epoch:

  Micro-meteoroid  — exponential pressure loss + tumble spin-up
  Power_Failure    — bus voltage collapse + heater failure (temp drop)
  Radiation_Death  — CPU latch-up (heartbeat=999) + thermal runaway

Output: Data/arb_telemetry_dataset.csv  (relative to project root)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Path resolution — relative to this file, works on any machine
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_telemetry(num_epochs: int = 2000, seed: int = None):
    """
    Generate a synthetic satellite telemetry dataset with one injected disaster.

    Args:
        num_epochs: Total number of one-minute simulation steps.
        seed:       Integer seed for fully reproducible output (None = random).

    Returns:
        df:            Complete telemetry DataFrame.
        anomaly_start: Epoch at which the disaster was injected.
        disaster_type: Name of the disaster scenario chosen.
    """
    rng = np.random.default_rng(seed)

    # Disaster window: starts no earlier than 20 % and no later than 85 %
    # into the mission, so the model always sees both nominal and anomalous phases.
    min_start     = int(num_epochs * 0.20)
    max_start     = int(num_epochs * 0.85)
    anomaly_start = int(rng.integers(min_start, max_start))
    anomaly_length = num_epochs - anomaly_start

    epochs = np.arange(1, num_epochs + 1)

    # Nominal sensor baselines (entire mission timeline)
    # Propellant drains slowly from 300 PSI → 280 PSI with Gaussian jitter
    propellant_psi = np.linspace(300.0, 280.0, num_epochs) + rng.normal(0, 0.5, num_epochs)

    # Bus voltage is stable at 28.5 V ± 0.2 V
    bus_voltage = rng.normal(28.5, 0.2, num_epochs)

    # CPU heartbeat latency sits between ~11 ms and ~19 ms (integer milliseconds)
    heartbeat_ms = rng.normal(15, 2, num_epochs).astype(int)

    # Gyroscope magnitude is near-zero in nominal attitude control (absolute value)
    gyro_mag = np.abs(rng.normal(0.01, 0.005, num_epochs))

    # ARB temperature fluctuates ±1 °C due to orbital sun/shadow cycling
    arb_temp_c = rng.normal(15.0, 1.0, num_epochs)

    # All epochs start as Nominal; anomalous epochs will be overwritten below
    target_status = np.array(['Nominal'] * num_epochs, dtype=object)

    # Inject one of three catastrophic failure scenarios
    disaster_type = rng.choice(['Micro-meteoroid', 'Power_Failure', 'Radiation_Death'])
    print(f"[GENERATOR] Scenario: {disaster_type}  |  Injected at epoch {anomaly_start}")

    if disaster_type == 'Micro-meteoroid':
        # Exponential pressure bleed + increasing tumble
        drop_curve = np.exp(np.linspace(0, 5, anomaly_length))
        propellant_psi[anomaly_start:] = np.clip(
            propellant_psi[anomaly_start:] - (drop_curve * 2), 0, None
        )
        gyro_mag[anomaly_start:] += np.linspace(0.5, 15.0, anomaly_length)

    elif disaster_type == 'Power_Failure':
        # Bus collapses to ~0.5 V; heaters lose power and temperature falls
        bus_voltage[anomaly_start:] = rng.normal(0.5, 0.1, anomaly_length)
        arb_temp_c[anomaly_start:]  -= np.linspace(2.0, 30.0, anomaly_length)

    elif disaster_type == 'Radiation_Death':
        # CPU latch-up: heartbeat locks to error sentinel 999
        # Battery attempts repeated reboots, driving thermal runaway
        heartbeat_ms[anomaly_start:] = 999
        arb_temp_c[anomaly_start:]   += np.linspace(5.0, 50.0, anomaly_length)

    target_status[anomaly_start:] = 'ANOMALY'

    # Assemble DataFrame and write to disk
    df = pd.DataFrame({
        'Epoch':               epochs,
        'Host_Propellant_PSI': np.round(propellant_psi, 2),
        'Host_Bus_Voltage':    np.round(bus_voltage, 2),
        'Host_Heartbeat_ms':   heartbeat_ms,
        'ARB_Gyro_Mag':        np.round(gyro_mag, 3),
        'ARB_Temp_C':          np.round(arb_temp_c, 2),
        'Target_Status':       target_status,
    })

    output_path = DATA_DIR / 'arb_telemetry_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"[GENERATOR] Dataset saved → '{output_path}'  ({num_epochs} epochs)")

    return df, anomaly_start, disaster_type


# Entry point
if __name__ == "__main__":
    # Fix seed=42 for fully reproducible hackathon demo runs.
    # Remove the seed argument (or pass seed=None) for randomised runs.
    df, strike_epoch, disaster = generate_telemetry(num_epochs=2000, seed=42)

    print(f"\n[GENERATOR] Sensor snapshot around the {disaster} failure point:")
    print(df.iloc[strike_epoch - 2: strike_epoch + 5].to_string(index=False))
