"""
ARB State Machine — Master OS Controller & Override Loop
Three-layer failsafe pipeline, executed in strict priority order:

  LAYER 1 — Coma Protocol
      24-hour dead-man's switch.  If the host CPU goes silent for
      dead_man_timeout_mins consecutive epochs, the ARB fires.

  LAYER 2 — Tactical Brain  (Isolation Forest)
      Multivariate structural-break detection with a graduated
      alert system (SUSPICIOUS → ARMED) and a 3-epoch debounce
      to suppress false alarms from natural sensor jitter.
      Uses continuous anomaly score (score_samples) for escalation.

  LAYER 3 — Strategic Brain  (Linear Regression)
      Proactive fuel-depletion forecast loaded from the frozen
      strategic_brain.pkl.  The burn is triggered proactive_burn_buffer
      epochs BEFORE the model predicts the 15 PSI breach, ensuring
      enough propellant remains to complete the disposal manoeuvre.

States

  NOMINAL ==> SUSPICIOUS ==> ARMED ==> PASSIVATION_FIRING ==> DEAD
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Path resolution — works from any machine, no absolute paths
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Data"
MODELS_DIR = Path(__file__).resolve().parent / "trained_models"


# 1. THE MASTER CONTROLLER
class ARBStateMachine:
    """
    Autonomous Retirement Blackbox finite-state machine.

    Parameters

    critical_fuel_limit : float
        PSI threshold below which a disposal burn is mandatory (default 15.0).
    dead_man_timeout_mins : int
        Consecutive silent epochs before Coma Protocol fires.
        Flight default: 1440 (24 hr).
        Sim testing with 2000-epoch datasets: set to ~60.
    mission_length_epochs : int
        Used to compute the calibration window (first 20 %).
    proactive_burn_buffer : int
        Epochs before the strategic forecast to issue the burn command.
        Gives propellant margin for the 11 m/s manoeuvre.
    suspicious_score_floor : float
        Anomaly score (from score_samples) below which a single anomalous
        epoch escalates to SUSPICIOUS rather than jumping straight to ARMED.
        Scores are negative; more negative = more anomalous.
    """

    def __init__(
        self,
        critical_fuel_limit:    float = 15.0,
        dead_man_timeout_mins:  int   = 1440,
        mission_length_epochs:  int   = 2000,
        proactive_burn_buffer:  int   = 100,
        suspicious_score_floor: float = -0.05,
    ):
        self.STATES        = ['NOMINAL', 'SUSPICIOUS', 'ARMED', 'PASSIVATION_FIRING', 'DEAD']
        self.current_state = 'NOMINAL'

        self.critical_fuel_limit    = critical_fuel_limit
        self.dead_man_timeout_mins  = dead_man_timeout_mins
        self.proactive_burn_buffer  = proactive_burn_buffer
        self.suspicious_score_floor = suspicious_score_floor

        # Calibration window: ignore anomaly logic for the first 20 % of the mission.
        # Derived from mission_length_epochs — no more magic numbers.
        self.calibration_window = int(mission_length_epochs * 0.20)

        # Runtime counters
        self.mins_since_last_heartbeat = 0
        self.tactical_anomaly_streak   = 0
        self.mission_epoch             = 0

        # Populated at boot by loading strategic_brain.pkl
        self.predicted_takeover_epoch: int | None = None

    def log(self, message: str):
        print(f"[EPOCH {self.mission_epoch:05d}] [{self.current_state:<20s}] {message}")

    def trigger_graveyard_burn(self, reason: str):
        """
        Initiate the irreversible 11 m/s apogee-raising disposal burn.

        In hardware, this asserts the GPIO line that drives the galvanically
        isolated relay coil.  time.sleep() is intentionally absent — a
        real-time flight OS must never block execution here.
        """
        self.current_state         = 'PASSIVATION_FIRING'
        self.tactical_anomaly_streak = 0  # clear counters on exit

        print("\n" + "!" * 65)
        self.log(f"PHYSICAL OVERRIDE ENGAGED — {reason}")
        self.log("Firing thruster relays from ARB supercapacitors...")
        self.log("Executing 11 m/s burn → 1,200 km graveyard shell.")
        print("!" * 65 + "\n")

        self.current_state = 'DEAD'
        self.log("Passivation complete. Fuel lines vented. Asset safely retired.")

    def process_telemetry_packet(
        self,
        psi:            float,
        bus_voltage:    float,
        heartbeat_ms:   int,
        tactical_signal: int,   # +1 nominal | -1 anomaly  (Isolation Forest binary)
        anomaly_score:  float,  # continuous score; more negative = more anomalous
    ) -> str:
        self.mission_epoch += 1

        # LAYER 1 — COMA PROTOCOL (24-Hour Dead-Man's Switch)
        # Heartbeat sentinel 999 = CPU latch-up / radio silence.
        # Any non-positive value is also treated as a lost ping.
        if heartbeat_ms == 999 or heartbeat_ms <= 0:
            self.mins_since_last_heartbeat += 1
        else:
            self.mins_since_last_heartbeat = 0  # valid ping resets the clock

        if self.mins_since_last_heartbeat >= self.dead_man_timeout_mins:
            self.trigger_graveyard_burn(
                "COMA PROTOCOL — total CPU/radio silence confirmed "
                f"({self.dead_man_timeout_mins} epochs without heartbeat)."
            )
            return self.current_state

        # LAYER 2 — TACTICAL BRAIN (Isolation Forest + debounce)
        if self.mission_epoch > self.calibration_window:

            if tactical_signal == -1:  # anomaly flagged
                self.tactical_anomaly_streak += 1

                # Graduated escalation driven by the continuous anomaly score:
                #   mild anomaly  (score near zero)  → SUSPICIOUS
                #   strong anomaly (very negative)   → ARMED immediately
                if self.current_state == 'NOMINAL':
                    if anomaly_score >= self.suspicious_score_floor:
                        self.current_state = 'SUSPICIOUS'
                        self.log(
                            f"Mild anomaly (score={anomaly_score:.4f}). "
                            "Entering SUSPICIOUS — monitoring."
                        )
                    else:
                        self.current_state = 'ARMED'
                        self.log(
                            f"Strong anomaly (score={anomaly_score:.4f}). "
                            "Arming relays — debouncing."
                        )

                elif self.current_state == 'SUSPICIOUS':
                    self.current_state = 'ARMED'
                    self.log(
                        f"Anomaly persists (score={anomaly_score:.4f}). "
                        "Escalating SUSPICIOUS → ARMED."
                    )

            else:  # clean reading — walk the alert level back down
                self.tactical_anomaly_streak = 0
                if self.current_state in ('ARMED', 'SUSPICIOUS'):
                    self.current_state = 'NOMINAL'
                    self.log("Sensor readings nominal. Disarming relays.")

            # Debounce trigger: 3 consecutive anomalous epochs confirm real event
            if self.tactical_anomaly_streak >= 3:
                self.trigger_graveyard_burn(
                    "TACTICAL TRAUMA — sustained multivariate structural chaos "
                    f"verified ({self.tactical_anomaly_streak} consecutive anomalous epochs)."
                )
                return self.current_state

        # LAYER 3 — STRATEGIC BRAIN (Proactive Fuel-Depletion Forecast)
        if self.predicted_takeover_epoch is not None:
            # Fire the burn when we're within proactive_burn_buffer epochs of
            # the model's predicted 15-PSI breach — not when we actually hit it.
            epochs_until_depletion = self.predicted_takeover_epoch - self.mission_epoch
            if epochs_until_depletion <= self.proactive_burn_buffer:
                self.trigger_graveyard_burn(
                    f"STRATEGIC DEPLETION — model forecasts 15 PSI breach at epoch "
                    f"{self.predicted_takeover_epoch} "
                    f"({epochs_until_depletion} epochs away, inside "
                    f"{self.proactive_burn_buffer}-epoch safety buffer)."
                )
                return self.current_state
        else:
            # Safety net: if strategic_brain.pkl failed to load, fall back to
            # a reactive hard threshold.  This path should never be reached in
            # a properly configured deployment.
            if psi <= self.critical_fuel_limit:
                self.trigger_graveyard_burn(
                    f"STRATEGIC DEPLETION [threshold fallback] — "
                    f"PSI={psi:.2f} breached {self.critical_fuel_limit} limit."
                )
                return self.current_state

        return self.current_state


# 2. LIVE DATASTREAM INTEGRATION LOOP
if __name__ == "__main__":

    print("=" * 65)
    print("  ARB Physical Master Controller OS — Booting")
    print("=" * 65)

    controller = ARBStateMachine(
        critical_fuel_limit   = 15.0,
        dead_man_timeout_mins = 1440,   # 24 hr — set to 60 for 2000-epoch sim tests
        mission_length_epochs = 2000,
        proactive_burn_buffer = 100,
        suspicious_score_floor = -0.05,
    )

    # Load telemetry dataset
    telemetry_path = DATA_DIR / 'arb_telemetry_dataset.csv'
    try:
        df = pd.read_csv(telemetry_path)
        print(f"[BOOT] Telemetry feed hooked: {len(df)} epochs — '{telemetry_path}'")
    except FileNotFoundError:
        print(f"[ERROR] Telemetry not found at '{telemetry_path}'. Run telemetry_generator.py first.")
        raise SystemExit(1)

    # Load all three frozen AI brains
    print("[BOOT] Loading frozen AI brains from storage chips...")
    try:
        scaler          = joblib.load(MODELS_DIR / "sensor_scaler.pkl")
        tactical_brain  = joblib.load(MODELS_DIR / "tactical_brain.pkl")
        strategic_brain = joblib.load(MODELS_DIR / "strategic_brain.pkl")
        print("[BOOT] Scaler, Tactical Brain, and Strategic Brain loaded into RAM.")
    except FileNotFoundError as e:
        print(f"[ERROR] Model file missing: {e}")
        print("[ERROR] Run the training notebooks first, then retry.")
        raise SystemExit(1)

    # Resolve strategic forecast at boot (algebraic solve: x = (y - b) / m)
    drain_rate     = strategic_brain.coef_[0]        # PSI lost per epoch (negative)
    initial_psi    = strategic_brain.intercept_       # extrapolated PSI at epoch 0
    takeover_epoch = int(
        (controller.critical_fuel_limit - initial_psi) / drain_rate
    )
    controller.predicted_takeover_epoch = takeover_epoch

    scheduled_burn_epoch = takeover_epoch - controller.proactive_burn_buffer
    print(
        f"[STRATEGIC] Predicted 15-PSI breach:  epoch {takeover_epoch}\n"
        f"[STRATEGIC] Proactive burn scheduled: epoch {scheduled_burn_epoch} "
        f"({controller.proactive_burn_buffer}-epoch safety buffer)"
    )

    # Pre-scale the entire telemetry feed and run both AI predictions
    FEATURES = [
        'Host_Propellant_PSI', 'Host_Bus_Voltage',
        'Host_Heartbeat_ms',   'ARB_Gyro_Mag', 'ARB_Temp_C',
    ]
    X_scaled = scaler.transform(df[FEATURES])

    # Binary signal (+1 / -1) for the debounce state machine
    df['Tactical_Signal'] = tactical_brain.predict(X_scaled)

    # Continuous anomaly score for graduated alert escalation
    # More negative = further from the nominal cluster = more anomalous
    df['Anomaly_Score'] = tactical_brain.score_samples(X_scaled)

    # Flight loop
    print("\n" + "-" * 65)
    print("  INITIATING LIVE AUTONOMOUS FLIGHT CONTROLLER LOOP")
    print("-" * 65 + "\n")

    for _, row in df.iterrows():
        state = controller.process_telemetry_packet(
            psi             = float(row['Host_Propellant_PSI']),
            bus_voltage     = float(row['Host_Bus_Voltage']),
            heartbeat_ms    = int(row['Host_Heartbeat_ms']),
            tactical_signal = int(row['Tactical_Signal']),
            anomaly_score   = float(row['Anomaly_Score']),
        )
        if state == 'DEAD':
            break

    print("\n[FLIGHT LOOP] Controller terminated.")
