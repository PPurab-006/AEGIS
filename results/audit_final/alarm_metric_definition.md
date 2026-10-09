# Definitive Scientific Definition: Alarm Rate Metrics in AEGIS

## 1. Executive Summary & Root Cause of Prior Ambiguity

In earlier reports and manuscripts, values such as **43.4% duty cycle** and **124 alarms/min** were reported side-by-side without clear units or mathematical definitions, causing confusion over how an alarm could fire 124 times per minute while simultaneously occupying nearly half of flight time.

Code inspection of `scripts/audit/t5_event_early_warning.py` (lines 161–165, 237–240) reveals the precise operational meaning:
1. **Duty Cycle (`duty_cycle_pct`)**: Measures the **percentage of discrete evaluated frames where the alarm output is active ($1$)**.
   $$\text{Duty Cycle} = \frac{\sum_{t=1}^N \mathbb{I}(\text{alarm}_t = 1)}{N} \times 100\%$$
2. **Rising Edge Rate (`alarm_rising_edges_per_min`)**: Formerly labeled ambiguously as `alarms_per_min`. It measures the **number of discrete $0 \to 1$ transitions per minute of active flight time**.
   $$\text{Rising Edge Rate} = \frac{\sum_{t=1}^N \mathbb{I}(\text{alarm}_t = 1 \land \text{alarm}_{t-1} = 0)}{T_{\text{minutes}}}$$
3. **False-Alarm Rising Edge Rate (`false_alarm_rising_edges_per_min`)**: The number of rising edges per minute that are **NOT** followed by a failure episode onset within the lookahead window $[t_{\text{edge}}, t_{\text{edge}} + W]$.

Because telemetry operates at **30 Hz (1,800 frames per minute)**, a model oscillating rapidly between 0 and 1 produces dozens of transitions per minute. A duty cycle of 43.4% means the alarm is active for **781 frames per minute (~26.0 seconds per minute)**, divided into approximately **124 distinct active pulse bursts per minute**.

## 2. Mathematical Formalization

Let the discrete time series for a flight be $t \in \{1, \dots, N\}$ sampled at $f_s = 30\,\text{Hz}$. Active flight duration is $T = N / (30 \times 60)$ minutes.

| Metric Name in Code | Mathematical Formulation | Physical Interpretation | Authoritative Final Paper Label |
| :--- | :--- | :--- | :--- |
| `duty_cycle_pct` | $\frac{1}{N} \sum_{t=1}^N a_t \times 100$ | Fraction of total flight duration alarm is active | **Alarm Duty Cycle (%)** |
| `alarm_rising_edges_per_min` | $\frac{1}{T} \sum_{t=2}^N a_t (1 - a_{t-1})$ | Frequency of newly triggered alarm bursts | **Alarm Rising Edges / min** |
| `false_alarm_rising_edges_per_min` | $\frac{1}{T} \sum_{t \in \text{edges}} \mathbb{I}(\text{no onset in } [t, t+W])$ | Rate of spurious alarm bursts without impending failure | **Spurious Alarm Edges / min** |

## 3. Discrepancy Reconciliation in Previous Prose

In `results/audit/t5_report.md` Section 4, the prose stated:
> "At native thresholds, the Frozen MLP generates ~38.7 alarms per minute and has a 43.7% duty cycle... false-alarm episodes occur at 27-32 episodes per minute. At matched 30% duty cycle, Retrained V0 achieves 57.3% early warnings, while HistGradientBoosting achieves 58.9%, and pure Yaw-Rate threshold achieves 51.2%."

**Forensic Audit Findings**:
1. **The ~38.7 alarms/min figure** was taken from an uncalibrated test baseline or scratch script, whereas the authoritative test table in the exact same document showed **124.0 rising edges/min** for Frozen MLP and **107.17 rising edges/min** for Retrained V0.
2. **The 57.3% / 58.9% / 51.2% figures** were **frame-level recalls** from the T2 baseline table at 30% duty cycle, NOT event-level early-warning percentages! The actual event-level early warning percentages at 30% duty cycle ($W=15$) are:
   - Frozen MLP: **77.28%** (any active) / **62.29%** (rising edge)
   - Retrained V0: **62.91%** (any active) / **44.98%** (rising edge)
   - HistGradientBoosting: **76.82%** (any active) / **63.52%** (rising edge)
   - Yaw Rate Threshold: **34.93%** (any active) / **15.46%** (rising edge)

The final paper must NEVER conflate frame-level recall with event-level anticipation, and must NEVER use the bare phrase "alarms per minute" without specifying "alarm rising edges per minute" or "duty cycle".
