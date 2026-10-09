#!/usr/bin/env python3
# SUPERSEDED: contains outdated claims; see docs/REPRODUCIBILITY.md and the published preprint.
"""
Complete AEGIS Research Preprint Document Generator.
Generates:
  docs/AEGIS_Preprint.docx
Converts to:
  docs/AEGIS_Preprint.pdf
Adheres strictly to the scientific source of truth and author instructions.
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from manuscript_framework import ManuscriptBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "results" / "figures" / "preprint"


def build_manuscript():
    print("Initializing ManuscriptBuilder...")
    builder = ManuscriptBuilder()

    # =========================================================================
    # TITLE & METADATA
    # =========================================================================
    builder.add_title("AEGIS: Telemetry-Based Early Warning of Monocular Visual Odometry Tracking Dropouts During Aggressive UAV Flight")
    builder.add_subtitle("A Methodological and Forensic Investigation into Anticipatory Telemetry Representation, Contemporaneous Feature Leakage, and Operational Alarm Constraints")
    builder.add_authors(
        "Purab Sen",
        "Autonomous Systems and Robotics Research Laboratory\nDepartment of Computer Science and Engineering"
    )

    # =========================================================================
    # ABSTRACT & KEYWORDS
    # =========================================================================
    abstract_text = (
        "Monocular Visual Odometry (VO) estimators are prone to abrupt tracking loss during aggressive unmanned aerial vehicle "
        "(UAV) maneuvers due to rapid rotational optical flow, motion blur, and feature matching collapse. Traditional failure "
        "mitigation relies on reactive re-detection or computationally intensive visual-inertial sensor fusion. In this work, "
        "we investigate AEGIS, a predictive monitoring framework that anticipates tracking dropouts ahead of time using exclusively "
        "lightweight vehicle kinematic and visual tracking telemetry (body yaw rate, optical flow displacement history, and rotational "
        "step indicators) sampled at 30 Hz. Through an exhaustive forensic audit across a 42-flight parametric sweep (31,615 active frames, "
        "51,041 raw frames) partitioned into strictly disjoint flight-level sets (15 train, 13 validation, 14 held-out test), we uncover "
        "and resolve a critical methodological flaw in prior evaluations: legacy target formulations included the contemporaneous frame, "
        "enabling models to exploit near-instantaneous optical flow collapse (P(failure | flow < 0.5 px) = 97.08%) to inflate reported "
        "discrimination (legacy AUROC 0.8046, AUPRC 0.7260). When formulated as a strict anticipatory task predicting strictly future "
        "failures across horizons K = 5 frames (165 ms) on 9,758 clean non-failed test frames, true predictive capacity remains substantially "
        "above chance: frozen and retrained Multi-Layer Perceptrons achieve strict AUROC 0.7665 and AUPRC 0.5982 (positive base rate 0.3024), "
        "while standard tree ensembles (HistGradientBoosting) achieve AUROC 0.7733 and AUPRC 0.6101, disproving claims of neural architecture "
        "superiority. Feature ablations demonstrate that while removing contemporaneous flow leaves strict performance intact (AUROC 0.7669), "
        "retaining optical flow history is vital (+0.0470 AUROC over pure kinematics). Furthermore, forensic trace analysis reveals that "
        "99.59% of failure episodes are transient single-frame (33.3 ms) tracking dropouts caused by feature re-detection resets rather "
        "than prolonged estimator drift. In event-level temporal evaluation (W = 15 frames / 0.50 s), 83.15% of failure episodes are preceded "
        "by active early warnings with a median lead time of 0.367 s; however, native operating thresholds incur a severe operational alarm burden "
        "(41.16% duty cycle, 107.17 alarm rising edges/min, 25.93 spurious edges/min). Domain generalization experiments confirm robust transfer "
        "across unseen trajectory geometries (LOGO AUROC 0.7688 ± 0.0319), but reveal severe vulnerability to held-out rotational velocities "
        "(LOYO AUROC 0.6466 ± 0.0423). Finally, multi-process ROS 2 Humble deployment across 2,991 frames at 30 Hz demonstrates zero message drops "
        "and deterministic end-to-end latency (median 1.20–1.51 ms, P99 1.62–2.00 ms, consuming < 6% of the frame period), while host stress pacing "
        "sustains zero message loss up to a nominal 500 Hz target (delivered rate ~206 Hz bounded by Linux timer resolution). We release complete "
        "causal replay logs, multi-seed artifacts, and audit ledgers to establish a rigorous methodological baseline for predictive robotic safety."
    )
    keywords_text = "Monocular Visual Odometry; Tracking Dropouts; Early Warning; Telemetry Representation; Contemporaneous Leakage; Flight Generalization; Robotic Middleware; ROS 2."
    builder.add_abstract(abstract_text, keywords_text)

    # =========================================================================
    # SECTION 1: INTRODUCTION
    # =========================================================================
    builder.add_heading1("1. Introduction")
    builder.add_paragraph(
        "Autonomous unmanned aerial vehicles (UAVs) operating in GPS-denied environments rely heavily on onboard vision-based state estimation. "
        "Monocular visual odometry (VO) and simultaneous localization and mapping (SLAM) pipelines—such as classical five-point Essential matrix "
        "estimators, semi-direct visual odometry (SVO), and direct sparse odometry (DSO)—provide an attractive, lightweight navigation solution "
        "for micro aerial platforms where payload and power constraints preclude heavy multi-sensor payloads (Nistér, 2004; Scaramuzza & Fraundorfer, "
        "2011; Forster et al., 2017). However, monocular estimators are intrinsically fragile during high-speed, agile flight. Rapid rotational "
        "maneuvers, aggressive yaw rotations, and sudden linear accelerations generate substantial inter-frame optical displacement, motion blur, "
        "and extreme parallax, causing Kanade-Lucas-Tomasi (KLT) feature tracking or descriptor matching to degenerate abruptly (Mur-Artal et al., 2015). "
        "When the number of valid geometric correspondences falls below the minimum required support, pose integration collapses, forcing the "
        "estimator into tracking failure."
    )
    builder.add_paragraph(
        "Conventionally, robotic perception stacks treat tracking loss as an unpredictable, reactive event. Upon detecting that feature support "
        "has collapsed, the pipeline initiates recovery routines: triggering global keyframe re-detection, freezing dead-reckoning integration, "
        "or falling back onto open-loop inertial extrapolation. In aggressive flight regimes, however, reactive mitigation often arrives too late. "
        "By the time tracking failure is officially acknowledged, image correspondences have dispersed, dead-reckoning errors compound quadratically, "
        "and flight control loops risk instability or catastrophic ground impact."
    )
    builder.add_paragraph(
        "To bridge this safety gap, predictive monitoring frameworks attempt to forecast imminent estimator degradation before geometric breakdown "
        "occurs. Prior work in early warning has explored deep spatio-temporal video models that process raw camera streams to identify visual blur "
        "or texture depletion. Yet, high-capacity convolutional or visual transformer architectures introduce prohibitive computational latency "
        "(frequently exceeding 20–50 ms per frame), creating an ironic operational paradox where the failure monitor consumes more CPU and GPU "
        "resources than the primary navigation pipeline itself."
    )
    builder.add_paragraph(
        "The AEGIS framework investigated in this study proposes an orthogonal, telemetry-driven paradigm: rather than performing expensive image-level "
        "inference, the system monitors lightweight, low-dimensional vehicle motion telemetry—specifically, onboard inertial measurement unit (IMU) "
        "body yaw rates, sparse KLT optical flow statistics, and rotational transition indicators. By evaluating these scalar signals across a causal, "
        "rolling history buffer (N = 6 frames, covering 165 ms at 30 Hz), AEGIS aims to anticipate monocular tracking dropouts K = 5 frames (~165 ms) "
        "into the future using a compact Multi-Layer Perceptron (SmallMLP, 1,057 parameters) or standard tree ensembles, executing in sub-millisecond latency."
    )
    builder.add_paragraph(
        "Crucially, this preprint presents the comprehensive forensic revision and empirical hardening of the AEGIS methodology. An earlier iteration "
        "of this research reported an optimistic test AUROC of 0.8046 and claimed neural architecture superiority alongside prolonged estimator collapse "
        "anticipation. As established in our formal forensic audit, that legacy evaluation was contaminated by contemporaneous feature leakage: the prediction "
        "target included the current frame, allowing models to exploit near-instantaneous optical flow collapse as a co-occurring detection signal. "
        "In this work, we rectify the target formulation to a strict future-only definition, purge ongoing failure frames, benchmark classical tabular "
        "models, characterize the true single-frame dropout failure mechanism, audit operational false alarm burden, and test domain generalization bounds."
    )

    # Subsection 1.1
    builder.add_heading2("1.1 Research Questions and Hypotheses")
    builder.add_paragraph(
        "To rigorously delineate the capabilities and limits of telemetry-based failure anticipation, this study addresses six central research questions:"
    )
    builder.add_paragraph(
        "• RQ1 (Anticipatory Telemetry Feasibility): Does onboard vehicle motion telemetry contain genuine predictive information regarding imminent "
        "monocular tracking dropouts under strict future-only evaluation, or was earlier performance entirely an artifact of contemporaneous leakage?\n"
        "  - Hypothesis H1: Telemetry retains above-chance discrimination (AUROC > 0.50, AUPRC > base rate) under strict future-only prediction.\n"
        "  - Hypothesis H0 (Null): Excluding contemporaneous cues reduces discrimination to chance level (AUROC ≈ 0.50)."
    )
    builder.add_paragraph(
        "• RQ2 (Model Architecture & Neural Necessity): Does a non-linear neural network (SmallMLP) confer an architectural advantage over classical "
        "tabular baselines, such as gradient-boosted decision trees (HistGradientBoosting) or linear logistic regression?\n"
        "  - Hypothesis H2: A compact Multi-Layer Perceptron outperforms classical machine learning estimators on tabular telemetry histories."
    )
    builder.add_paragraph(
        "• RQ3 (Feature Modality Contributions): Does predictive capability depend upon contemporaneous lag-0 optical flow, or does historical flow and "
        "kinematic telemetry provide sufficient anticipatory signal?\n"
        "  - Hypothesis H3: Dropping contemporaneous lag-0 flow preserves discrimination, whereas dropping optical flow entirely causes significant degradation."
    )
    builder.add_paragraph(
        "• RQ4 (Domain Generalization Boundaries): How reliably does the learned predictor generalize across unseen flight conditions, specifically "
        "evaluating Leave-One-Cell-Out (LOCO), Leave-One-Yaw-Rate-Out (LOYO), and Leave-One-Geometry-Out (LOGO)?\n"
        "  - Hypothesis H4: Telemetry representations generalize robustly across unseen trajectory geometries but experience degradation under held-out yaw rates."
    )
    builder.add_paragraph(
        "• RQ5 (Operational Utility vs. Alarm Burden): Does statistical early warning translate into practical operational utility, or does high alarm "
        "duty cycle and false-alarm frequency create prohibitive alarm fatigue?\n"
        "  - Hypothesis H5: High advance warning rates are accompanied by an operationally substantial alarm burden at uncalibrated native thresholds."
    )
    builder.add_paragraph(
        "• RQ6 (Real-Time Systems Feasibility): Can causal streaming inference operate deterministically within a multi-process ROS 2 middleware stack "
        "without message loss under nominal and high-rate stress conditions?"
    )

    # =========================================================================
    # SECTION 2: RELATED WORK
    # =========================================================================
    builder.add_heading1("2. Related Work")
    builder.add_heading2("2.1 Visual Odometry and Failure Modes in Agile UAV Flight")
    builder.add_paragraph(
        "Monocular visual odometry algorithms estimate the six-degree-of-freedom (6-DoF) ego-motion of a camera by tracking photometric or geometric "
        "features across consecutive video frames (Nistér, 2004; Scaramuzza & Fraundorfer, 2011). In sparse feature-based approaches, such as ORB-SLAM3 "
        "(Campos et al., 2021) or classical five-point Essential matrix pipelines with RANSAC outlier rejection, the accuracy and numerical stability "
        "of pose recovery depend directly on maintaining a well-distributed set of inlier correspondences across the image plane. Under aggressive UAV flight, "
        "however, angular velocities frequently exceed several hundred degrees per second, inducing severe motion blur, large inter-frame pixel displacements, "
        "and rapid perspective distortions that exceed the convergence basin of KLT optical flow trackers (Lucas & Kanade, 1981; Shi & Tomasi, 1994). "
        "When inlier counts drop below theoretical minimal solvers (e.g., 5 points for relative rotation and translation up to scale, or 8 points for normalized "
        "linear systems), pose estimation cannot be solved, resulting in tracking loss."
    )
    builder.add_paragraph(
        "While visual-inertial odometry (VIO) frameworks—such as VINS-Mono (Qin et al., 2018) and OKVIS (Leutenegger et al., 2015)—fuse high-rate IMU "
        "measurements to bridge visual tracking outages, severe rotational acceleration can still cause estimator divergence if visual updates fail "
        "repeatedly or if IMU biases drift during prolonged visual starvation. Understanding and anticipating tracking dropouts is therefore vital "
        "even in fused navigation systems."
    )

    builder.add_heading2("2.2 Failure Detection, Quality Estimation, and Uncertainty in SLAM")
    builder.add_paragraph(
        "Prior literature in SLAM reliability has largely focused on retrospective quality assessment and failure detection. Techniques such as covariance "
        "intersection, residual monitoring, and pose graph optimization verification assess whether an existing state estimate is degenerate (Kaess et al., "
        "2012). Recent learning-based methods have trained convolutional neural networks to predict tracking uncertainty, optical flow confidence, or optical "
        "degradation from raw imagery (Ran et al., 2017; Costante et al., 2016). For example, Loquercio et al. (2021) explored learning-based perception "
        "monitoring for autonomous drone racing. However, vision-based neural networks impose severe computational overhead, often requiring GPU acceleration "
        "and introducing tens of milliseconds of latency, rendering them poorly suited for micro-aerial platforms."
    )

    builder.add_heading2("2.3 Predictive Safety Filtering and Telemetry-Based Monitoring")
    builder.add_paragraph(
        "Predictive monitoring shifts the focus from retrospective detection to anticipatory forecasting. In safety-critical control, Control Barrier "
        "Functions (CBFs) and reachability analysis establish forward-invariant safe sets to prevent dynamical state boundary violations (Ames et al., 2019). "
        "In perception-aware navigation, researchers have investigated trajectory generation that explicitly optimizes visual feature visibility and avoids "
        "rotational velocities known to induce motion blur (Falanga et al., 2018). However, such methods require full analytical models of the environment "
        "and perception pipeline. Telemetry-based empirical monitoring represents an alternative paradigm: passively sampling available vehicle flight dynamics "
        "and low-level visual tracking metrics to identify precursors of tracking breakdown without explicit 3D scene reconstruction."
    )

    builder.add_heading2("2.4 Methodological Traps: The Fallacy of Contemporaneous Feature Leakage")
    builder.add_paragraph(
        "A recognized vulnerability in predictive time-series modeling is temporal and contemporaneous feature leakage (Kaufman et al., 2012; Wiens et al., "
        "2019). Contemporaneous leakage occurs when a predictor ostensibly trained to forecast future events incorporates features or target windows that "
        "overlap with the onset of the event itself. In medical informatics, Wiens et al. (2019) demonstrated how clinical alarm models frequently achieve "
        "near-perfect AUROC simply by detecting symptoms of ongoing deterioration rather than forecasting onset. In robotics, similar traps occur when "
        "low-level tracking indicators (such as optical flow velocity) collapse synchronously with estimator failure; if the current frame is included in the "
        "prediction window, the model functions as a synchronous failure detector rather than a true anticipator. Rigorous evaluation necessitates strictly "
        "future-offset labels and explicit purging of currently failed frames."
    )

    # =========================================================================
    # SECTION 3: METHODS
    # =========================================================================
    builder.add_heading1("3. Methods")
    builder.add_heading2("3.1 Simulation Environment and Visual Odometry Pipeline")
    builder.add_paragraph(
        "To acquire systematic, multi-condition flight data with ground-truth state telemetry, we deployed a high-fidelity Software-In-The-Loop (SITL) "
        "simulation environment coupling the PX4 Autopilot (v1.14) with Gazebo Sim (Garden). The simulated platform was a Holybro x500 quadrotor "
        "operating within a realistic textured agricultural terrain (`agriculture.world`). The platform was equipped with an onboard downward-forward "
        "monocular camera oriented at 45° pitch, streaming 640 × 480 grayscale imagery at 30 frames per second (FPS), tightly synchronized with onboard "
        "odometry, IMU gyroscopes, and a classical monocular visual odometry node (`minimal_vo.py`)."
    )
    builder.add_paragraph(
        "The visual odometry pipeline implements a classical sparse feature-based tracking architecture. Salient corners are identified using the Shi-Tomasi "
        "`goodFeaturesToTrack` detector (up to 2,000 corners, quality level 0.001, minimum distance 5 pixels). Point correspondences between consecutive frames "
        "are tracked using pyramidal KLT optical flow (Lucas-Kanade with 3 pyramid levels, window size 21 × 21 pixels). From matched correspondences, relative "
        "pose is estimated via OpenCV's 5-point Essential matrix algorithm (`findEssentialMat`) with RANSAC outlier rejection (threshold 1.0 pixel, confidence "
        "0.999), followed by `recoverPose` to resolve the rotation matrix and translation direction up to scale."
    )

    builder.add_heading2("3.2 Dataset Sweep Design and Quarantined Partitioning")
    builder.add_paragraph(
        "To induce a broad spectrum of tracking challenges spanning nominal cruising to catastrophic feature starvation, we executed a parametric "
        "4 × 4 flight sweep combining four commanded body yaw rates across four distinct trajectory geometries, with three independent flight repetitions "
        "per cell (48 planned flights):"
    )
    builder.add_paragraph(
        "• Commanded Yaw Rates: Gentle (G, 7.5°/s), Moderate (M, 30.0°/s), Aggressive (A, 60.0°/s), Extreme (E, 90.0°/s).\n"
        "• Trajectory Geometries: Circle (C, radius 10 m), Box / Figure-8 (B, 15 m span), Star / Spiral (S, 12 m span), Helix / Handheld (H, variable radius)."
    )
    builder.add_paragraph(
        "To ensure flight safety and physical validity, strict physical envelope gating was applied across all recorded flight logs. Flights exhibiting "
        "takeoff crashes, sustained ceiling violations (z > 25 m), or failsafe aborts were disqualified. Exactly 6 flights were excluded: `sweep_G_B_R1`, "
        "`sweep_G_H_R1`, `sweep_G_H_R2`, `sweep_G_H_R3`, `sweep_M_H_R2`, and `sweep_M_H_R3` (the Gentle-Helix cell was entirely unviable). "
        "The resulting dataset contains exactly 42 eligible autonomous flights totaling 51,041 raw recorded frames. Active flight frames were strictly "
        "quarantined by enforcing an in-flight altitude threshold (z ≥ 2.0 m), yielding 31,615 active flight frames."
    )
    builder.add_paragraph(
        "To guarantee zero optimization or temporal leakage between splits, flights were partitioned strictly at the flight level using a fixed pseudorandom "
        "seed (Seed 42):"
    )
    builder.add_paragraph(
        "• Training Split: 15 flights (11,328 active frames, 35.8% of dataset)\n"
        "• Validation Split: 13 flights (9,736 active frames, 30.8% of dataset)\n"
        "• Held-Out Test Split: 14 flights (10,551 active frames, 33.4% of dataset)\n"
        "• Set Overlap: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅ (exactly zero flight or frame overlap)."
    )

    # Insert Figure 1
    builder.add_figure(
        FIG_DIR / "fig1_pipeline_and_methodology.png",
        "End-to-end research methodology, data flow, and multi-tier verification framework. "
        "(a) Parametric 4 × 4 flight sweep combining 4 yaw rates and 4 trajectory geometries in PX4/Gazebo SITL, filtered by physical envelope gates into "
        "42 eligible flights (31,615 active frames) and partitioned strictly at the flight level into disjoint Train (15), Validation (13), and Test (14) sets. "
        "(b) Streaming causal feature extraction: 30 Hz passive telemetry (yaw rate ω_z, optical flow displacement v̄_flow, rotational indicator is_r) is buffered "
        "in an N = 6 FIFO ring buffer to construct a 15-dimensional lagged feature vector. Predictions are evaluated against the strict target y_strict(t) predicting "
        "future failure across [t+1, t+5] with ongoing failures purged. (c) Three-tier forensic verification spanning strict offline discrimination, "
        "multi-domain generalization (LOCO/LOYO/LOGO), event-level early-warning dynamics, and multi-process ROS 2 Humble runtime validation."
    )

    # Table 1
    t1_headers = ["Split Partition", "Flight Count", "Raw Frames", "Active Frames (z ≥ 2m)", "Failure Frames", "Base Failure Rate (%)", "Failure Episodes"]
    t1_data = [
        ["Train Split", "15 flights", "18,252", "11,328", "705", "6.22%", "701"],
        ["Validation Split", "13 flights", "15,820", "9,736", "620", "6.37%", "617"],
        ["Held-Out Test Split", "14 flights", "16,969", "10,551", "653", "6.19%", "651"],
        ["Total / Pooled", "42 flights", "51,041", "31,615", "1,978", "6.26%", "1,969"]
    ]
    builder.add_table_with_caption(
        "Parametric Flight Sweep Dataset Composition and Strict Flight-Level Partitioning. "
        "Quarantined by physical flight envelope gates; zero flight or frame overlap exists across splits.",
        t1_headers, t1_data, [1.3, 0.8, 0.8, 1.2, 0.9, 1.0, 0.9]
    )

    builder.add_heading2("3.3 Visual Odometry Tracking Failure Definition")
    builder.add_paragraph(
        "In classical monocular VO, the Essential matrix solver requires at least 5 point correspondences to resolve relative motion, but standard "
        "implementations require a higher support threshold to guard against degenerate planar configurations and collinear feature sets. In `minimal_vo.py`, "
        "the tracking state at frame t is governed by the inlier count of the recovered pose:"
    )
    builder.add_equation("Failure(t) = 𝟙( num_inliers_pose(t) < 8 )")
    builder.add_paragraph(
        "Across the complete 31,615 active flight frames, exactly 1,978 frames satisfied this failure condition, establishing a dataset-wide frame-level "
        "failure rate of 6.2565%. When consecutive failure frames are grouped into contiguous failure episodes, the dataset contains exactly 1,969 distinct "
        "failure episodes."
    )

    builder.add_heading2("3.4 Legacy Target Formulation and Strict Anticipatory Definition")
    builder.add_paragraph(
        "In the legacy AEGIS implementation, the early-warning target was constructed across a forward lookahead window of K = 5 frames (~165 ms):"
    )
    builder.add_equation("y_legacy(t) = max( F_t, F_{t+1}, F_{t+2}, ..., F_{t+K} )")
    builder.add_paragraph(
        "Notice that Equation (2) includes the current frame F_t. Because the model is evaluated on frame t, if F_t = 1 (tracking has already collapsed), "
        "y_legacy(t) evaluates to 1. As revealed in our forensic audit, this created severe contemporaneous feature leakage: the model was rewarded for "
        "detecting co-occurring tracking failure in the present frame rather than forecasting future breakdown. To restore strict scientific validity, "
        "we define the strict anticipatory prediction target:"
    )
    builder.add_equation("y_strict(t) = max( F_{t+1}, F_{t+2}, ..., F_{t+K} )")
    builder.add_paragraph(
        "Crucially, under strict evaluation, all frames where visual odometry is already failing (F_t = 1) must be strictly excluded from the evaluation set. "
        "In our 14 held-out test flights (10,551 active frames), dropping 5 warmup frames (due to backward lag buffering) and 5 tail frames (due to forward "
        "lookahead boundaries) yields 10,411 evaluated frames in the legacy protocol. Among these, exactly 653 frames have F_t = 1. Purging these ongoing "
        "failures leaves exactly 9,758 clean strict test frames, with 2,951 strict positive labels (positive-class base rate = 0.302418)."
    )

    builder.add_heading2("3.5 Causal Telemetry History Buffer and Feature Construction")
    builder.add_paragraph(
        "To ensure strictly causal operation without future temporal access, the predictor maintains a rolling FIFO ring buffer of size N = 6 steps. "
        "At each frame t, the buffer ingests three scalar signals streamed from onboard flight telemetry at 30 Hz:"
    )
    builder.add_paragraph(
        "1. `eis_yaw_rate_deg`: Platform rotational yaw velocity (|ω_z|, °/s) derived from IMU gyro telemetry.\n"
        "2. `feature_vel_mean`: Mean 2D pixel displacement magnitude (px/frame) of tracked KLT features between consecutive frames.\n"
        "3. `is_r_frame`: Binary indicator signaling elevated rotational rate. (Forensic audit confirmed is_r ≡ (|ω_z| > 15.0°/s) with 100% deterministic identity)."
    )
    builder.add_paragraph(
        "From this rolling history, a 15-dimensional lagged feature vector x_t is constructed using backward lag indices l ∈ {0, 1, 2, 3, 5}, corresponding "
        "to temporal offsets of 0 ms, 33.3 ms, 66.7 ms, 100.0 ms, and 166.7 ms prior to the current frame:"
    )
    builder.add_equation("x(t) = [ ω_z(t - l),  v̄_flow(t - l),  is_r(t - l) ]_{l ∈ {0, 1, 2, 3, 5}} ∈ ℝ¹⁵")
    builder.add_paragraph(
        "All features are normalized using a frozen `StandardScaler` whose mean and variance parameters were fitted exclusively on the 15 training flights "
        "(11,328 frames) and applied without modification during testing."
    )

    builder.add_heading2("3.6 Predictor Architectures and Training Setup")
    builder.add_paragraph(
        "The primary neural model investigated is `SmallMLP`, a compact feed-forward Multi-Layer Perceptron designed for sub-millisecond onboard inference. "
        "The network architecture comprises an input layer (d_in = 15), a first hidden layer of 32 units with ReLU activation, a second hidden layer of 16 units "
        "with ReLU activation, and a single linear output unit mapped through a sigmoid activation to produce failure probability P(y(t) = 1 | x_t). "
        "The network contains exactly 1,057 trainable parameters."
    )
    builder.add_paragraph(
        "The model was trained using Binary Cross-Entropy with Logits loss (`BCEWithLogitsLoss`) optimized via Adam (learning rate 10^-3, batch size 64, "
        "weight decay 10^-4) for up to 50 epochs with early stopping based on validation loss (patience 10 epochs). In the retrained strict configuration "
        "(Variant V0), models were trained exclusively on clean non-failed training frames to predict y_strict(t). The operational decision threshold "
        "θ* was tuned exclusively on the validation partition by maximizing F1 score, yielding θ* = 0.54."
    )

    builder.add_heading2("3.7 Baseline Models and Systematic Feature Ablations")
    builder.add_paragraph(
        "To rigorously benchmark SmallMLP, we evaluated four baseline estimators on the identical strict test set (9,758 frames):"
    )
    builder.add_paragraph(
        "• HistGradientBoostingClassifier (HGB): Scikit-learn's gradient-boosted decision tree ensemble (max_iter=100, learning_rate=0.1, min_samples_leaf=20).\n"
        "• LogisticRegression (LR): L2-regularized linear model (C=1.0) serving as a linear baseline.\n"
        "• Yaw-Rate Threshold: A heuristic classifier triggering an alarm whenever current body yaw rate |ω_z| exceeds a threshold tuned on validation data (33.5°/s).\n"
        "• Frames-Since-Dropout Recency: A baseline triggering an alarm if a failure occurred recently, testing temporal autoregressive clustering."
    )
    builder.add_paragraph(
        "To dissect feature importance, we executed a systematic ablation study across five retrained model variants:"
    )
    builder.add_paragraph(
        "• V0 (All 15 Features): Full baseline vector.\n"
        "• V1 (Drop Flow Lag-0, 14 Features): Removes contemporaneous optical flow v̄_flow(t), relying on lags {1, 2, 3, 5} and all yaw features.\n"
        "• V2 (Drop All Flow, 10 Features): Removes optical flow entirely, retaining only yaw rate and is_r lags.\n"
        "• V3 (Yaw Lags Only, 5 Features): Retains exclusively raw yaw rate lags, removing flow and the redundant is_r flag.\n"
        "• V4 (Flow Lags Only, 5 Features): Retains exclusively optical flow lags, removing all kinematic gyro telemetry."
    )

    builder.add_heading2("3.8 Cross-Domain Generalization Protocols")
    builder.add_paragraph(
        "Standard test splits stratify by flight repetition (R3 held out), meaning that sibling flights (R1, R2) conducted under identical trajectory cells "
        "appear in the training split. To assess true domain transfer, we executed three cross-validation protocols:"
    )
    builder.add_paragraph(
        "1. Leave-One-Cell-Out (LOCO, 15 Folds): Iteratively holds out each specific parameter cell (e.g., `A_C`) while training on the remaining 14 cells.\n"
        "2. Leave-One-Yaw-Rate-Out (LOYO, 4 Folds): Holds out entire rotational regimes (Gentle, Moderate, Aggressive, Extreme), testing extrapolation across speed.\n"
        "3. Leave-One-Geometry-Out (LOGO, 4 Folds): Holds out entire geometric flight paths (Circle, Box, Star, Helix), testing geometric invariance."
    )

    builder.add_heading2("3.9 Event-Level Early-Warning Metrics")
    builder.add_paragraph(
        "Frame-level classification metrics (AUROC, AUPRC) do not fully reflect operational utility in continuous flight. We evaluate event-level early-warning "
        "dynamics across pre-onset anticipation windows W ∈ {5, 10, 15, 30} frames (corresponding to 0.167 s, 0.333 s, 0.500 s, and 1.000 s at 30 Hz). "
        "For each failure episode onset t_onset (where F_{t_onset-1} = 0 and F_{t_onset} = 1), we evaluate:"
    )
    builder.add_paragraph(
        "• Rising-Edge Early Warning: An alarm transitions from 0 to 1 within [t_onset - W, t_onset - 1].\n"
        "• Any-Active Early Warning: An alarm is high at any point within [t_onset - W, t_onset - 1], including alarms active prior to entering the window.\n"
        "• Advance Lead Time: The temporal separation Δt = (t_onset - t_alarm) / 30 between the earliest alarm and failure onset.\n"
        "• Operational Alarm Burden: Characterized by Alarm Duty Cycle (% of frames alarm is active), Alarm Rising Edges per minute (rate of distinct alarm bursts), "
        "and Spurious Rising Edges per minute (alarms not followed by a failure onset within W frames)."
    )

    builder.add_heading2("3.10 Statistical Robustness and Inference")
    builder.add_paragraph(
        "To guard against split sampling variance, we implemented two statistical uncertainty protocols: (1) Multi-seed model training across 5 random seeds "
        "(0 to 4) on the strict task; and (2) Non-parametric flight-level bootstrap resampling with 2,000 repetitions. In each bootstrap repetition, 14 test "
        "flights were sampled with replacement, and paired metric differences (e.g., Legacy minus Strict, Frozen vs. Retrained V0) were computed. Two-sided "
        "empirical p-values were evaluated as p = 2 · min(P(Δ > 0), P(Δ < 0))."
    )

    builder.add_heading2("3.11 Causal Sequential Replay")
    builder.add_paragraph(
        "To verify that offline batch inference does not benefit from non-causal batch normalization or lookahead artifacts, we executed causal sequential "
        "streaming replay across all 14 test flights in exact chronological timestamp order. Telemetry was fed frame-by-frame into a standalone ring-buffer "
        "predictor class, and predictions were compared to batch PyTorch outputs at single-precision floating-point resolution."
    )

    builder.add_heading2("3.12 Multi-Process ROS 2 Middleware Validation")
    builder.add_paragraph(
        "To evaluate real-time feasibility in robotic middleware, we constructed an independent three-process ROS 2 Humble deployment harness: "
        "(1) Telemetry Publisher Process: Streams serialized `sensor_msgs/msg/Imu` and custom telemetry messages across `/telemetry/motion`; "
        "(2) Predictor Process: Executes `StreamingFailurePredictor` inside an `rclpy` node, subscribing to `/telemetry/motion` and publishing "
        "predictions to `/vo/failure_prediction`; and (3) Diagnostics Receiver Process: Records monotonic clock timestamps. "
        "We executed Run A (30 Hz real flight telemetry across 4 flights) and Run B (multi-rate stress testing at 60, 100, 200, and 500 Hz nominal pacing)."
    )

    # =========================================================================
    # SECTION 4: RESULTS
    # =========================================================================
    builder.add_heading1("4. Results")
    builder.add_heading2("4.1 Forensic Investigation and Correction of Legacy Evaluation")
    builder.add_paragraph(
        "The legacy evaluation reported an impressive test AUROC of 0.8046 and AUPRC of 0.7260. However, our forensic audit establishes that this performance "
        "was substantially inflated by contemporaneous feature leakage. In the legacy test set (10,411 frames), visual odometry was actively failing on 653 frames. "
        "On these frames, KLT tracking breakdown caused `feature_vel_mean_lag0` to drop below 0.5 px with probability 97.32%, and P(failure | flow < 0.5 px) was 97.08%. "
        "The frozen SmallMLP scored positive (p ≥ 0.50) on 642 of these 653 frames (98.32%), mechanically adding 642 artificial True Positives and inflating legacy "
        "True Positives from 2,002 to 2,644."
    )
    builder.add_paragraph(
        "When ongoing failures are purged and the target is restricted strictly to future frames [t+1, t+5], the frozen model's AUROC drops from 0.8046 to 0.7665 "
        "(a decline of -0.0381), while AUPRC drops from 0.7260 to 0.5977 (a decline of -0.1283). In paired flight-level bootstrap testing (2,000 resamples), "
        "the mean legacy-versus-strict AUROC difference is +0.0387 (95% CI [0.0308, 0.0478], p < 0.001), and the mean AUPRC difference is +0.1289 (95% CI [0.1087, 0.1569], "
        "p < 0.001). This confirms that contemporaneous leakage produced statistically significant, non-trivial performance inflation."
    )

    # Insert Figure 2
    builder.add_figure(
        FIG_DIR / "fig2_legacy_vs_strict_performance.png",
        "Empirical quantification of contemporaneous feature leakage bias on the 14-flight held-out test partition. "
        "(a) Receiver Operating Characteristic (ROC) curves: Legacy evaluation (AUROC 0.8046, red solid) versus strict future-only evaluation on the frozen model "
        "(AUROC 0.7665, navy dashed) and retrained V0 (AUROC 0.7665, teal dash-dot). (b) Precision-Recall (PR) trajectories: Legacy evaluation (AUPRC 0.7260, base rate 0.3462) "
        "versus strict evaluation (AUPRC 0.5977 / 0.5982, base rate 0.3024). (c) Paired bootstrap difference distributions (Legacy minus Strict, 2,000 resamples): "
        "Leakage statistically inflated AUROC by +0.0387 (95% CI [0.0308, 0.0478], p < 0.001) and AUPRC by +0.1289 (95% CI [0.1087, 0.1569], p < 0.001)."
    )

    builder.add_paragraph(
        "The physical mechanism underlying this contemporaneous leakage is confirmed by forensic trace analysis of tracking failure events (Figure 3). "
        "During aggressive rotational maneuvers, feature tracking breakdown causes instantaneous optical flow displacement to collapse below 0.5 px "
        "on 97.32% of failure frames. Because the legacy evaluation included ongoing failure frames (F_t = 1) in its target window, legacy models "
        "exploited this concurrent optical flow collapse as an instantaneous failure flag, yielding artificial True Positives rather than genuine anticipation. "
        "Furthermore, empirical analysis of all 1,969 failure episodes across the 42 flights demonstrates that 99.59% are single-frame tracking skips (33.3 ms duration), "
        "refuting prior assumptions of prolonged visual odometry divergence."
    )

    # Insert Figure 3
    builder.add_figure(
        FIG_DIR / "fig3_forensic_failure_mechanism.png",
        "Forensic evidence of visual odometry failure mechanism and contemporaneous feature leakage. "
        "(a) Empirical conditional probabilities of optical flow collapse (< 0.5 px) and tracking failure (num_inliers_pose < 8), demonstrating 97.3% co-occurrence. "
        "(b) Empirical episode duration distribution across 1,969 failure episodes: 99.59% are single-frame (33.3 ms) tracking skips. "
        "(c) Micro-episode onset profile: Inlier count collapses to 0.06 at failure onset (t = 0) and rebounds immediately to ~1,259 inliers at t = +1 "
        "as Shi-Tomasi feature re-detection re-initializes tracking."
    )

    builder.add_heading2("4.2 Strict Predictive Performance")
    builder.add_paragraph(
        "Despite the removal of contemporaneous leakage, the strict evaluation confirms that passive flight telemetry contains statistically genuine anticipatory "
        "information. Under strict future-only prediction, both the frozen SmallMLP and retrained strict SmallMLP V0 achieve an identical test AUROC of 0.7665 "
        "(AUPRC 0.5977 and 0.5982, respectively), substantially outperforming the 0.50 chance baseline and the 0.3024 positive base rate. "
        "At the validation-selected optimal threshold θ* = 0.54, retrained SmallMLP V0 operates at Precision 0.5085, Recall 0.6682, and F1 score 0.5775."
    )

    builder.add_heading2("4.3 Baseline Benchmarks and Feature Modality Ablations")
    builder.add_paragraph(
        "A critical claim of the original manuscript was that neural network architectures (MLP) were uniquely capable of learning failure dynamics. "
        "As reported in Table 2 and Figure 4, benchmarking on the strict task directly refutes this claim. A standard gradient-boosted decision tree ensemble "
        "(HistGradientBoosting) achieves a strict AUROC of 0.7733, an AUPRC of 0.6101, and an F1 score of 0.5896 (at validation-tuned threshold θ = 0.34), "
        "outperforming SmallMLP across all discrimination metrics (+0.0068 AUROC, +0.0119 AUPRC). Linear logistic regression achieves AUROC 0.7161 and AUPRC 0.5361, "
        "while a zero-parameter heuristic threshold on body yaw rate (|ω_z| > 33.5°/s) achieves AUROC 0.6895 and AUPRC 0.4709."
    )

    # Insert Figure 4
    builder.add_figure(
        FIG_DIR / "fig4_strict_model_comparisons.png",
        "Strict benchmark comparison across predictive models on the 14 held-out test flights (N = 9,758 frames). "
        "(a) Offline discrimination metrics (AUROC and AUPRC): HistGradientBoosting (0.7733 / 0.6101) outperforms SmallMLP V0 (0.7665 / 0.5982), "
        "Logistic Regression (0.7161 / 0.5361), and heuristic yaw rate thresholds (0.6895 / 0.4709). (b) Detection recall evaluated at a matched 30% alarm duty cycle: "
        "HistGradientBoosting (55.91%) and SmallMLP V0 (56.05%) achieve comparable sensitivity, while outperforming pure kinematics (49.41%) and chance (28.36%)."
    )

    # Table 2
    t2_headers = ["Model", "Protocol", "Threshold (θ)", "AUROC", "AUPRC", "F1 Score", "Precision", "Recall", "Accuracy", "Alarm Duty"]
    t2_data = [
        ["HistGradientBoosting", "Strict Clean", "0.34 (val)", "0.7733", "0.6101", "0.5896", "0.5305", "0.6635", "72.06%", "37.83%"],
        ["HistGradientBoosting", "Strict Clean", "0.50 (fix)", "0.7733", "0.6101", "0.5068", "0.6141", "0.4314", "74.61%", "21.24%"],
        ["SmallMLP (Retrained V0)", "Strict Clean", "0.54 (val)", "0.7665", "0.5982", "0.5775", "0.5085", "0.6682", "70.43%", "39.74%"],
        ["SmallMLP (Retrained V0)", "Strict Clean", "0.50 (fix)", "0.7665", "0.5982", "0.5758", "0.4687", "0.7462", "66.75%", "48.15%"],
        ["SmallMLP (Frozen Strict)", "Strict Clean", "0.50 (fix)", "0.7665", "0.5977", "0.5836", "0.5120", "0.6784", "70.72%", "40.07%"],
        ["Logistic Regression", "Strict Clean", "0.27 (val)", "0.7161", "0.5361", "0.5462", "0.4468", "0.7025", "64.70%", "47.55%"],
        ["Yaw-Rate Threshold", "Strict Clean", "33.5°/s", "0.6895", "0.4709", "0.5393", "0.4332", "0.7143", "63.10%", "49.87%"],
        ["Dropout Recency", "Strict Clean", "0.01 (val)", "0.3964", "0.2972", "0.1282", "0.4032", "0.0762", "68.65%", "5.72%"],
        ["[Comparator] Frozen Legacy", "Legacy (Leak)", "0.50 (fix)", "0.8046", "0.7260", "0.6484", "0.5808", "0.7336", "72.45%", "43.72%"]
    ]
    builder.add_table_with_caption(
        "Primary Benchmark: Discrimination and Classification Metrics on Strict Held-Out Test Flights. "
        "Evaluated on N = 9,758 clean non-failed test frames (positive base rate = 0.3024). Legacy frozen model shown strictly as an audit comparator.",
        t2_headers, t2_data, [1.35, 0.65, 0.65, 0.52, 0.52, 0.52, 0.54, 0.52, 0.58, 0.65]
    )

    builder.add_paragraph(
        "Systematic feature ablations (Table 3, Figure 5) elucidate the physical basis of early warning. Removing contemporaneous optical flow (Variant V1) "
        "yields AUROC 0.7669 and AUPRC 0.6002, matching or marginally exceeding full Variant V0 (AUROC 0.7665). This proves that the strict anticipatory signal "
        "does not rely on lag-0 flow disclosure. However, removing all optical flow features (Variant V2) causes AUROC to drop sharply to 0.7199 and AUPRC to 0.4911 "
        "(-0.0466 AUROC, -0.1071 AUPRC). Relying exclusively on flow lags (V4, AUROC 0.7519) yields stronger anticipation than relying exclusively on yaw lags "
        "(V3, AUROC 0.7267). Thus, historical optical flow dynamics provide essential, non-redundant predictive structure."
    )

    # Insert Figure 5
    builder.add_figure(
        FIG_DIR / "fig5_feature_ablations.png",
        "Systematic feature ablation analysis across retrained model variants on the strict prediction task [t+1, t+5]. "
        "Variant V0 (All 15 features: AUROC 0.7665, AUPRC 0.5982); Variant V1 (Drop lag-0 flow: AUROC 0.7669, AUPRC 0.6002) confirms that contemporaneous flow "
        "is redundant for strict prediction; Variant V2 (Drop all flow: AUROC 0.7199, AUPRC 0.4911) demonstrates a severe ~0.047 AUROC penalty when visual dynamics "
        "are removed; Variant V3 (Yaw only: AUROC 0.7267); Variant V4 (Flow only: AUROC 0.7519) demonstrates that flow history is more informative than gyro rates."
    )

    # Table 3
    t3_headers = ["Ablation Variant", "Feat Count", "Included Modalities", "Threshold (θ)", "AUROC", "AUPRC", "F1 Score", "Precision", "Recall", "Δ AUROC (vs V0)"]
    t3_data = [
        ["V0: Full Baseline", "15", "Yaw + Flow + is_r (Lags 0-5)", "0.54 (val)", "0.7665", "0.5982", "0.5775", "0.5085", "0.6682", "Reference"],
        ["V1: Drop Flow Lag-0", "14", "Yaw (0-5) + Flow (1-5) + is_r", "0.56 (val)", "0.7669", "0.6002", "0.5809", "0.5427", "0.6249", "+0.0004"],
        ["V2: Drop All Flow", "10", "Yaw (0-5) + is_r (0-5)", "0.44 (val)", "0.7199", "0.4911", "0.5517", "0.4237", "0.7906", "-0.0466"],
        ["V3: Yaw Lags Only", "5", "Pure Yaw Rate (Lags 0-5)", "0.43 (val)", "0.7267", "0.4959", "0.5603", "0.4353", "0.7862", "-0.0398"],
        ["V4: Flow Lags Only", "5", "Pure Optical Flow (Lags 0-5)", "0.48 (val)", "0.7519", "0.5680", "0.5721", "0.4856", "0.6960", "-0.0146"]
    ]
    builder.add_table_with_caption(
        "Systematic Feature Modality Ablations on the Strict Anticipatory Task [t+1, t+5]. "
        "Evaluated across retrained SmallMLP architectures at validation-tuned decision thresholds.",
        t3_headers, t3_data, [1.15, 0.45, 1.35, 0.55, 0.48, 0.48, 0.48, 0.48, 0.48, 0.60]
    )

    builder.add_heading2("4.4 Cross-Domain Generalization Bounds")
    builder.add_paragraph(
        "The standard repeat-held-out test split evaluates flights whose sibling repetitions (e.g., R1, R2) appeared during training. "
        "To test true domain transfer, Table 4 and Figure 6 report cross-validation results across unseen parameter cells, yaw rates, and flight paths. "
        "Leave-One-Cell-Out (LOCO, 15 folds) achieves a mean AUROC of 0.6980 ± 0.0355 (range 0.6415 to 0.7470). Leave-One-Geometry-Out (LOGO, 4 folds) "
        "maintains strong performance, achieving mean AUROC 0.7688 ± 0.0319 (Circle: 0.7827, Box: 0.7410, Star: 0.7761, Helix: 0.7202), comparable to "
        "the standard test split."
    )
    builder.add_paragraph(
        "In contrast, Leave-One-Yaw-Rate-Out (LOYO, 4 folds) exhibits severe performance collapse, dropping to a mean AUROC of 0.6466 ± 0.0423. "
        "When moderate yaw rates (30°/s) are held out, AUROC falls to 0.5949—approaching random chance. This empirical collapse disproves any claim "
        "that AEGIS generalizes across unseen rotational velocities: the model learns rotational velocity operating thresholds and cannot extrapolate "
        "outside its training envelope."
    )

    # Insert Figure 6
    builder.add_figure(
        FIG_DIR / "fig6_cross_domain_generalization.png",
        "Cross-domain generalization bounds across operational flight dimensions. "
        "(a) Leave-One-Cell-Out (LOCO, 15 folds): Mean AUROC = 0.6980 ± 0.0355. (b) Leave-One-Yaw-Rate-Out (LOYO, 4 folds): Severe collapse across rotational regimes, "
        "dropping to mean AUROC = 0.6466 ± 0.0423 (Moderate yaw collapses to 0.5949). (c) Leave-One-Geometry-Out (LOGO, 4 folds): Robust invariance across flight path "
        "shapes, maintaining mean AUROC = 0.7688 ± 0.0319 (range 0.7342 to 0.8109)."
    )

    # Table 4
    t4_headers = ["Generalization Regime", "Folds", "Target Tested", "AUROC (Mean)", "AUROC (Std)", "AUROC (Min)", "AUROC (Max)", "AUPRC (Mean)", "F1 Score"]
    t4_data = [
        ["Standard Repeat-Held-Out", "1", "Unseen Executions (R3)", "0.7669", "—", "0.7669", "0.7669", "0.6003", "0.5834"],
        ["Leave-One-Cell-Out (LOCO)", "15", "Held-Out Trajectory Cell", "0.6980", "± 0.0355", "0.6415", "0.7470", "0.5240", "0.4212"],
        ["Leave-One-Yaw-Rate-Out (LOYO)", "4", "Held-Out Yaw Rate Bin", "0.6466", "± 0.0423", "0.5949", "0.6947", "0.4527", "0.4214"],
        ["Leave-One-Geometry-Out (LOGO)", "4", "Held-Out Flight Geometry", "0.7688", "± 0.0319", "0.7342", "0.8109", "0.6106", "0.5274"]
    ]
    builder.add_table_with_caption(
        "Cross-Domain Generalization Benchmarks Across Operational Dimensions. "
        "Demonstrates strong geometric path invariance alongside severe rotational velocity extrapolation failure.",
        t4_headers, t4_data, [1.35, 0.45, 1.20, 0.58, 0.58, 0.58, 0.58, 0.58, 0.60]
    )

    builder.add_heading2("4.5 Event-Level Early-Warning Dynamics and Operational Alarm Burden")
    builder.add_paragraph(
        "Evaluating continuous flight performance across temporal pre-onset windows W ∈ {5, 10, 15, 30} frames (Table 5, Figure 7) reveals that "
        "telemetry-based early warning genuinely precedes failure onset. For retrained SmallMLP V0 at W = 15 frames (0.50 s pre-onset window), "
        "59.35% of failure episodes are preceded by a clean alarm rising edge (rising-edge median lead 0.367 s, mean lead 0.337 s), and 83.15% of "
        "episodes are preceded by an active alarm state (any-active median lead 0.500 s, mean lead 0.459 s). For HistGradientBoosting, 76.66% exhibit "
        "rising-edge early warnings with rising-edge median lead 0.367 s and mean lead 0.331 s, while 88.72% exhibit active alarms with any-active median "
        "lead 0.500 s and mean lead 0.443 s. The divergence between rising-edge and any-active lead times stems directly from high baseline duty cycles "
        "(40–41%): under the any-active metric, alarms already asserted prior to the anticipation window are window-bounded at t_onset - W (0.50 s), "
        "extending nominal lead times toward the window ceiling. At W = 30 frames (1.00 s), warning coverage expands to 85.94% (rising-edge median lead 0.767 s, "
        "mean lead 0.697 s)."
    )
    builder.add_paragraph(
        "Crucially, however, our audit uncovers the severe operational alarm burden imposed by native operating thresholds. At θ* = 0.54, retrained SmallMLP V0 "
        "maintains an alarm duty cycle of 41.16%, triggering 107.17 alarm rising edges per minute, of which 25.93 per minute are spurious alarms (not followed by "
        "a failure within 0.50 s). This corresponds to a false alarm burst every 2.3 seconds of flight time. When operating thresholds are calibrated to a "
        "constrained 30% duty cycle, false alarms decrease to 7.73 rising edges/min, but rising-edge early warning falls to 62.91% (Table 5)."
    )

    # Insert Figure 7
    builder.add_figure(
        FIG_DIR / "fig7_event_early_warning_and_alarm_burden.png",
        "Event-level temporal early-warning dynamics and operational alarm burden. "
        "(a) Anticipation window sweep W ∈ {5, 10, 15, 30} frames: Rising-edge early-warning rates (28.7% to 77.9%) versus any-active alarm rates (80.5% to 85.9%). "
        "(b) Advance lead-time distribution at W = 15 frames (0.50 s horizon): Median lead time of 0.367 s (mean 0.337 s). "
        "(c) Operational alarm burden trade-off at native thresholds: High alarm duty cycle (41.16%), elevated rising edge frequency (107.17 edges/min), "
        "and substantial false alarm rate (25.93 spurious bursts/min) illustrate severe alarm fatigue constraints."
    )

    # Table 5
    t5_headers = ["Model Evaluated", "Window", "Threshold (θ)", "Duty Cycle", "Rising (/min)", "False (/min)", "Rising Early", "Active Early", "Rising Med Lead", "Rising Mean Lead"]
    t5_data = [
        ["SmallMLP V0 (Native)", "W=15 (0.50s)", "0.54", "41.16%", "107.17", "25.93", "59.35%", "83.15%", "0.367 s", "0.337 s"],
        ["HistGradientBoosting", "W=15 (0.50s)", "0.34", "40.03%", "122.11", "20.44", "76.66%", "88.72%", "0.367 s", "0.331 s"],
        ["Frozen SmallMLP", "W=15 (0.50s)", "0.50", "43.44%", "124.00", "21.30", "62.29%", "87.79%", "0.367 s", "0.336 s"],
        ["SmallMLP V0 (Native)", "W=30 (1.00s)", "0.54", "41.16%", "107.17", "7.73", "77.90%", "85.94%", "0.767 s", "0.697 s"],
        ["SmallMLP V0 (30% Calib)", "W=15 (0.50s)", "0.64", "30.01%", "75.22", "7.73", "62.91%", "72.18%", "0.333 s", "0.315 s"],
        ["HGB (30% Calib)", "W=15 (0.50s)", "0.42", "30.01%", "118.16", "11.33", "79.13%", "83.46%", "0.367 s", "0.336 s"],
        ["Yaw Rate (30% Calib)", "W=15 (0.50s)", "52.0°/s", "30.01%", "60.45", "12.19", "34.93%", "69.86%", "0.350 s", "0.319 s"]
    ]
    builder.add_table_with_caption(
        "Event-Level Early-Warning Performance and Operational Alarm Burden Across Telemetry Predictors. "
        "Evaluated on held-out test flight failure episodes. Distinguishes clean rising-edge transitions from pre-existing active alarm states. "
        "Note: Reported lead times reflect the initial rising edge (0→1 transition) within the W=15 window. Under the any-active definition "
        "(including pre-existing high-state alarms window-bounded at t_onset - W = 0.50 s), any-active median and mean lead times are "
        "0.500 s / 0.459 s for SmallMLP V0, and 0.500 s / 0.443 s for HistGradientBoosting.",
        t5_headers, t5_data, [1.30, 0.62, 0.52, 0.52, 0.58, 0.56, 0.56, 0.56, 0.74, 0.74]
    )

    builder.add_heading2("4.6 Micro-Episode Failure Mechanism Analysis")
    builder.add_paragraph(
        "Earlier descriptions characterized VO tracking failures as prolonged estimator collapse or divergent trajectory drift. "
        "Our forensic trace analysis of `minimal_vo.py` directly refutes this framing. As illustrated in Figure 3, tracking breakdown triggers "
        "an algorithmic re-detection loop: when `num_matched < 8`, the estimator clears its point cache (`self.prev_pts = None`). In the immediate "
        "subsequent frame, `goodFeaturesToTrack` re-detects corners across the raw image, synchronously emitting 0 inliers, 0 optical flow, and skipping "
        "pose integration for exactly one sample. In the next frame, hundreds of new features are successfully tracked, restoring tracking."
    )
    builder.add_paragraph(
        "Empirically, across all 1,969 failure episodes in the dataset, exactly 1,961 episodes (99.5937%) are exactly 1 frame long (33.3 ms duration). "
        "Only 7 episodes span 2 frames (0.3555%), and exactly 1 episode spans 3 frames (0.0508%). The maximum failure duration observed across all 42 flights "
        "is 100 ms. The benchmark therefore represents transient monocular tracking dropouts / single-frame pose-update skips, not multi-second estimator divergence."
    )

    builder.add_heading2("4.7 Causal Sequential Replay Parity")
    builder.add_paragraph(
        "Causal streaming replay was executed across all 14 test flights in exact chronological order (10,411 evaluated frames; reconciles a legacy "
        "discrepancy of 70 frames caused by NaN-skipping tail boundary truncation). Comparing streaming FIFO ring-buffer predictions against batch "
        "PyTorch inference yielded a maximum absolute probability difference of 4.17 × 10^-7 and a mean difference of 3.23 × 10^-8 across all frames. "
        "This establishes numerical floating-point parity between streaming causal inference and offline models."
    )

    builder.add_heading2("4.8 Multi-Process ROS 2 Middleware Validation and Stress Profiling")
    builder.add_paragraph(
        "To test real-time robotics feasibility, AEGIS was deployed across three independent OS processes in ROS 2 Humble. In Run A (30 Hz real flight telemetry "
        "across 4 test flights, 2,991 total frames), zero message drops were observed (0.00% drop rate). Predictor callback compute time exhibited a median of "
        "0.53–0.69 ms (P99 0.79–1.01 ms). End-to-end round-trip latency (from telemetry publication to diagnosis receipt) had a median of 1.20–1.51 ms "
        "(P99 1.62–2.00 ms), consuming less than 6.0% of the 33.3 ms frame period (Table 6, Figure 8)."
    )
    builder.add_paragraph(
        "In Run B (multi-rate stress sweep through nominal rates of 60, 100, 200, and 500 Hz), exactly 400 of 400 messages were delivered in each condition "
        "with zero message loss. However, actual delivered burst rates were bounded by host Linux non-real-time timer sleep granularity (~206 Hz at the 500 Hz target). "
        "While zero drops occurred through 500 Hz nominal pacing on the host testbed, embedded flight computer validation remains an open requirement."
    )

    # Insert Figure 8
    builder.add_figure(
        FIG_DIR / "fig8_streaming_replay_and_ros2_runtime.png",
        "Causal streaming replay numerical parity and multi-process ROS 2 Humble runtime verification. "
        "(a) Absolute error distribution between streaming FIFO ring-buffer inference and offline PyTorch batch evaluation across 10,411 frames (max difference 4.17 × 10^-7). "
        "(b) ROS 2 multi-process latency breakdown at 30 Hz across 2,991 frames: Middleware transport (median ~0.41 ms), callback compute (median ~0.60 ms, P99 ~0.88 ms), "
        "and total end-to-end round-trip latency (median ~1.36 ms, P99 ~1.84 ms, consuming < 6% of the 33.3 ms frame period). "
        "(c) Stress pacing sweep across nominal rates up to 500 Hz: Zero message drops observed, with delivered rates bounded to ~206 Hz by Linux timer resolution."
    )

    # Table 6
    t6_headers = ["Flight / Condition", "Rate", "Sent", "Recv", "Drop", "Transport", "Compute (Med)", "Compute (P99)", "Round-Trip (Med)", "Round-Trip (P99)"]
    t6_data = [
        ["sweep_A_C_R1 (Aggressive)", "30 Hz", "759", "759", "0.00%", "0.36 ms", "0.56 ms", "0.83 ms", "1.26 ms", "1.81 ms"],
        ["sweep_G_C_R1 (Gentle)", "30 Hz", "745", "745", "0.00%", "0.46 ms", "0.69 ms", "1.01 ms", "1.51 ms", "2.00 ms"],
        ["sweep_E_C_R1 (Extreme)", "30 Hz", "738", "738", "0.00%", "0.46 ms", "0.63 ms", "0.90 ms", "1.44 ms", "1.95 ms"],
        ["sweep_M_C_R1 (Moderate)", "30 Hz", "749", "749", "0.00%", "0.35 ms", "0.53 ms", "0.79 ms", "1.20 ms", "1.62 ms"],
        ["Stress Test: 60 Hz Nom.", "51.2 act", "400", "400", "0.00%", "—", "—", "—", "1.22 ms", "1.64 ms"],
        ["Stress Test: 100 Hz Nom.", "77.8 act", "400", "400", "0.00%", "—", "—", "—", "1.11 ms", "2.03 ms"],
        ["Stress Test: 200 Hz Nom.", "142.5 act", "400", "400", "0.00%", "—", "—", "—", "0.88 ms", "23.29 ms"],
        ["Stress Test: 500 Hz Nom.", "205.9 act", "400", "400", "0.00%", "—", "—", "—", "0.65 ms", "10.49 ms"]
    ]
    builder.add_table_with_caption(
        "Multi-Process ROS 2 Humble Runtime Latency Breakdown and High-Rate Stress Pacing. "
        "Measured across three independent operating system processes with true monotonic timestamp logging.",
        t6_headers, t6_data, [1.35, 0.45, 0.48, 0.48, 0.48, 0.58, 0.65, 0.65, 0.69, 0.69]
    )

    builder.add_heading2("4.9 Multi-Seed Robustness and Statistical Uncertainty")
    builder.add_paragraph(
        "Multi-seed training of SmallMLP V0 across 5 random seeds (0 to 4) on the strict task yields tight performance bounds: AUROC = 0.7681 ± 0.0024 "
        "(range [0.7656, 0.7707]), AUPRC = 0.6019 ± 0.0025, and F1 = 0.5808 ± 0.0032 (Table 7). In 2,000-repetition flight-level bootstrap testing, "
        "the mean AUROC for the frozen strict model is 0.7634 (95% CI [0.7140, 0.8053]) and for retrained SmallMLP V0 is 0.7631 (95% CI [0.7150, 0.8039]). "
        "The paired difference between frozen and retrained models is +0.0004 AUROC (p = 0.904), confirming that the model weights are stable and that the "
        "performance revision stems entirely from the corrected scientific definition of the task."
    )

    # Table 7
    t7_headers = ["Seed / Contrast", "Threshold (θ)", "AUROC", "AUPRC", "F1 Score", "Precision", "Recall", "Alarm Rate (%)"]
    t7_data = [
        ["Seed 0", "0.51 (val)", "0.7665", "0.6016", "0.5838", "0.4999", "0.7015", "42.44%"],
        ["Seed 1", "0.57 (val)", "0.7707", "0.6051", "0.5817", "0.5298", "0.6449", "36.81%"],
        ["Seed 2", "0.52 (val)", "0.7707", "0.6036", "0.5759", "0.4955", "0.6876", "41.97%"],
        ["Seed 3", "0.50 (val)", "0.7656", "0.5988", "0.5793", "0.4731", "0.7472", "47.77%"],
        ["Seed 4", "0.55 (val)", "0.7669", "0.6003", "0.5834", "0.5410", "0.6330", "35.39%"],
        ["Mean ± Std (5 Seeds)", "0.53 ± 0.03", "0.7681 ± 0.0024", "0.6019 ± 0.0025", "0.5808 ± 0.0032", "0.5078 ± 0.0274", "0.6828 ± 0.0459", "40.87 ± 4.94%"],
        ["Frozen Strict (Bootstrap)", "0.50 (fix)", "0.7634 [0.714, 0.805]", "0.5938 [0.497, 0.672]", "0.5772 [0.490, 0.651]", "0.5102 [0.432, 0.589]", "0.6708 [0.504, 0.799]", "40.07%"],
        ["Retrained V0 (Bootstrap)", "0.54 (val)", "0.7631 [0.715, 0.804]", "0.5944 [0.500, 0.669]", "0.5705 [0.477, 0.648]", "0.5062 [0.426, 0.584]", "0.6600 [0.486, 0.797]", "39.74%"],
        ["Paired Diff (Frozen - V0)", "—", "+0.0004 (p = 0.904)", "-0.0007 (p = 0.879)", "+0.0067 (p = 0.347)", "+0.0040 (p = 0.566)", "+0.0108 (p = 0.226)", "—"]
    ]
    builder.add_table_with_caption(
        "Multi-Seed Training Uncertainty and 2,000-Repetition Flight-Level Bootstrap Confidence Intervals. "
        "Confirms statistical equivalence between frozen strict and retrained V0 models (AUROC difference p = 0.904).",
        t7_headers, t7_data, [1.40, 0.65, 1.05, 0.95, 0.85, 0.55, 0.55, 0.50]
    )

    # =========================================================================
    # SECTION 5: DISCUSSION
    # =========================================================================
    builder.add_heading1("5. Discussion")
    builder.add_heading2("5.1 What the Strict Result Demonstrates")
    builder.add_paragraph(
        "The primary positive scientific finding of this study is that passive vehicle telemetry contains genuine anticipatory information "
        "concerning monocular visual odometry tracking dropouts. When contemporaneous failure frames are quarantined and models are evaluated strictly "
        "on predicting future tracking loss [t+1, t+5], discrimination remains substantially superior to random chance (AUROC 0.7665, AUPRC 0.5982 "
        "versus a 0.3024 baseline). This demonstrates that tracking failure during aggressive maneuvers is not a memoryless, white-noise event. "
        "Rather, rotational acceleration builds over hundreds of milliseconds, generating subtle kinematic signatures and optical flow deceleration "
        "patterns that reliably precede geometric collapse."
    )

    builder.add_heading2("5.2 Why the Legacy Result Was Methodologically Inflated")
    builder.add_paragraph(
        "Our forensic audit conclusively explains why the legacy result (AUROC 0.8046, AUPRC 0.7260) was artificially high. In `minimal_vo.py`, "
        "when tracking breaks down, optical flow cannot be tracked and synchronously returns 0.0 px. Because the legacy evaluation window evaluated "
        "max(F_t, ..., F_{t+5}), any frame where failure was already occurring was labeled positive. The model learned to detect this instantaneous "
        "zero-flow drop, effectively operating as a co-occurring failure detector on 653 test frames. This added 642 artificial true positives. "
        "The paired bootstrap delta (+0.0387 AUROC, +0.1289 AUPRC, p < 0.001) confirms that contemporaneous feature disclosure caused statistically "
        "significant inflation. This serves as an important cautionary case study for time-series evaluation in robotic perception."
    )

    builder.add_heading2("5.3 Architectural Lessons: Neural Networks vs. Decision Trees")
    builder.add_paragraph(
        "The finding that HistGradientBoosting matches or exceeds SmallMLP (AUROC 0.7733 vs. 0.7665; AUPRC 0.6101 vs. 0.5982) underscores an important "
        "lesson in applied machine learning: on compact, tabular telemetry vectors (15 dimensions), deep or multi-layer neural architectures confer no "
        "inherent advantage over well-tuned tree ensembles. Gradient-boosted decision trees naturally partition orthogonal kinematic thresholds and "
        "handle non-monotonic optical flow features without requiring gradient descent tuning or weight regularization. While SmallMLP remains viable "
        "for C++ deployment via TensorRT or ONNX Runtime, neural superiority is definitively refuted."
    )

    builder.add_heading2("5.4 Failure Mechanism Implications: Dropouts vs. Drift")
    builder.add_paragraph(
        "The empirical discovery that 99.59% of failure episodes are single-frame (33.3 ms) tracking skips redefines the operational problem. "
        "In this benchmark, the visual odometry pipeline does not experience multi-second estimator divergence where position estimates drift into infinity. "
        "Instead, the pipeline experiences a transient geometric hitch: feature points are lost, pose update is skipped for one frame, new corners are detected, "
        "and tracking resumes immediately. Therefore, early warning in this context should not be conceptualized as an alert for emergency vehicle landing; "
        "rather, it serves as a predictive cue for sensor fusion filtering—for instance, signaling an Extended Kalman Filter (EKF) to temporarily down-weight "
        "visual measurement updates and rely on IMU dead-reckoning for 1–2 samples."
    )

    builder.add_heading2("5.5 The Generalization Boundary: Geometry vs. Dynamics")
    builder.add_paragraph(
        "The contrast between LOGO (mean AUROC 0.7688) and LOYO (mean AUROC 0.6466) defines the generalization boundary of telemetry-based monitoring. "
        "The model is largely invariant to trajectory shape: whether flying circles, figure-8s, or stars, the physical relationship between yaw rate, "
        "optical flow, and feature loss remains stable. In contrast, the model fails to extrapolate across rotational velocities. When trained on gentle "
        "and aggressive yaw rates, it struggles to interpolate moderate regimes (AUROC 0.5949). The model does not learn a universal physical law of "
        "optical flow geometry; it learns decision boundaries calibrated to specific velocity distributions. Deploying such monitors in open-world autonomy "
        "requires training across dense, continuous dynamic sweeps."
    )

    builder.add_heading2("5.6 Statistical Anticipation vs. Operational Utility: The Alarm Burden")
    builder.add_paragraph(
        "Perhaps the most critical operational finding is the divergence between statistical early warning and practical usability. At native thresholds "
        "(θ* = 0.54), SmallMLP V0 warns ahead of 83.15% of failure episodes within 0.50 s. However, it incurs an alarm duty cycle of 41.16% and emits 25.93 "
        "false alarm bursts per minute (a spurious alarm every 2.3 seconds). In an unaugmented flight control architecture, such frequent alarms would induce "
        "severe alarm fatigue, causing the autopilot to constantly reject valid visual measurements. Calibrating to a 30% duty cycle reduces false alarms "
        "to 7.73/min, but lowers rising-edge early warnings to 62.91%. Practical deployment will require temporal hysteresis, multi-frame consensus filtering, "
        "or continuous variance scaling rather than raw binary thresholding."
    )

    builder.add_heading2("5.7 What the Study Does Not Establish")
    builder.add_paragraph(
        "To maintain strict scientific integrity, we explicitly demarcate the boundaries of this research:\n"
        "1. This study does NOT establish physical hardware validation. All experiments were conducted in PX4/Gazebo SITL simulation.\n"
        "2. This study does NOT establish closed-loop autonomous flight intervention. Predictions were logged passively without modifying flight trajectories.\n"
        "3. This study does NOT establish 500 Hz real-time capacity on embedded robotics computers. Zero message loss was observed up to 500 Hz nominal pacing "
        "on an Intel Core i9 desktop, but delivered rates were bounded to ~206 Hz by Linux timer resolution, and Jetson/embedded testing remains future work.\n"
        "4. This study does NOT establish robust out-of-distribution rotational extrapolation, as proven by the LOYO null result."
    )

    # =========================================================================
    # SECTION 6: LIMITATIONS
    # =========================================================================
    builder.add_heading1("6. Limitations")
    builder.add_paragraph(
        "Several methodological and technical limitations constrain the scope of this work:\n"
        "• Simulation Envelope: While Gazebo Sim provides realistic rigid-body dynamics and lighting, it lacks environmental optical artifacts such as "
        "direct solar glare, lens dirt, rolling-shutter distortion, and dynamic shadows present in physical outdoor UAV flight.\n"
        "• Monocular Pipeline Specificity: The benchmark evaluated a single classical five-point Essential matrix VO implementation (`minimal_vo.py`). "
        "Modern feature-based SLAM systems (e.g., ORB-SLAM3) and direct methods (e.g., DSO) incorporate multi-frame local bundle adjustment and keyframe caches "
        "that may exhibit different failure dynamics.\n"
        "• Prohibitive Native False Alarm Rate: The high frequency of spurious alarms (~26 bursts/min) prevents direct, unaugmented closed-loop flight control integration.\n"
        "• Lack of Cross-Platform Generalization: The model was evaluated on a single airframe (Holybro x500) and camera field of view (FOV). Variations in camera "
        "focal length or vehicle mass would alter optical flow scaling, requiring domain re-calibration."
    )

    # =========================================================================
    # SECTION 7: CONCLUSION
    # =========================================================================
    builder.add_heading1("7. Conclusion")
    builder.add_paragraph(
        "This study provides a rigorous, methodologically hardened evaluation of telemetry-based early warning for monocular visual odometry tracking dropouts "
        "during aggressive UAV flight. Through systematic forensic auditing, we resolved contemporaneous feature leakage, demonstrating that while the legacy "
        "AUROC of 0.8046 was inflated, passive kinematic and visual telemetry retains statistically genuine anticipatory capacity under strict future-only "
        "evaluation (AUROC 0.7665, AUPRC 0.5982). Standard tree ensembles (HistGradientBoosting, AUROC 0.7733) match or slightly outperform neural architectures, "
        "disproving claims of neural superiority. Forensic trace analysis revealed that tracking failures in this domain are transient single-frame dropouts "
        "(99.59% lasting 33.3 ms) rather than prolonged estimator divergence. While early warning precedes failure onsets by 0.3 to 0.5 seconds across >80% of episodes, "
        "operational utility is constrained by high alarm burden and sensitivity to unseen rotational velocities. Finally, multi-process ROS 2 middleware validation "
        "confirms deterministic sub-millisecond execution headroom (<6% of frame budget). In conclusion, passive telemetry represents a viable, computationally "
        "negligible predictive signal for robotic state estimation, provided operational false-alarm mitigation and velocity domain boundaries are strictly managed."
    )

    # =========================================================================
    # SECTION 8: REPRODUCIBILITY & DATA AVAILABILITY
    # =========================================================================
    builder.add_heading1("8. Data, Code, and Reproducibility")
    builder.add_paragraph(
        "All code, configuration files, trained model weights, evaluation logs, and audit ledgers are preserved in the research repository. "
        "The frozen model weights (`v0_strict_seed42.pt`, SHA-256: `bbbd9ce5020d9d46ae59da22dcf82bfae2efd09d7f98f0e482fbf24dec921f43`) and scaler "
        "(`v0_strict_scaler.joblib`, SHA-256: `b444e21481c5e28ae5308a4b659ca5b7e6c01d82b124e0c405241fdc051282a7`) allow exact numerical reproduction "
        "of all strict test metrics. Automated verification test suites (`pytest tests/test_causal_leakage.py`) verify zero future access and numerical "
        "streaming parity. All ledgers (`FINAL_NUMBER_LEDGER.csv`, `CLAIM_LEDGER.md`, `DISCREPANCY_LOG.md`) are archived in `results/audit_final/`."
    )

    # =========================================================================
    # REFERENCES
    # =========================================================================
    builder.add_heading1("References")
    references = [
        "Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., & Tabuada, P. (2019). Control barrier functions: Theory and applications. In 2019 18th European Control Conference (ECC) (pp. 3420-3431). IEEE.",
        "Campos, C., Elvira, R., Rodríguez, J. J. G., Montiel, J. M., & Tardós, J. D. (2021). ORB-SLAM3: An accurate open-source library for visual, visual–inertial, and multimap SLAM. IEEE Transactions on Robotics, 37(6), 1874-1890.",
        "Costante, G., Delmerico, J., Werlberger, M., & Scaramuzza, D. (2016). Perception-aware path planning for autonomous visual navigation. In 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 5258-5265). IEEE.",
        "Engel, J., Koltun, V., & Cremers, D. (2018). Direct sparse odometry. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(3), 611-625.",
        "Falanga, D., Foehn, P., Lu, P., & Scaramuzza, D. (2018). PAMPC: Perception-aware model predictive control for quadrotors. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 1-8). IEEE.",
        "Forster, C., Runz, M., Faessler, M., & Scaramuzza, D. (2017). SVO: Semi-direct visual odometry for monocular and multicamera systems. IEEE Transactions on Robotics, 33(2), 249-265.",
        "Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). iSAM2: Incremental smoothing and mapping using the Bayes tree. The International Journal of Robotics Research, 31(2), 216-235.",
        "Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data (TKDD), 6(4), 1-21.",
        "Leutenegger, S., Lynen, S., Bosse, M., Siegwart, R., & Furgale, P. (2015). Keyframe-based visual–inertial odometry using nonlinear optimization. The International Journal of Robotics Research, 34(3), 314-334.",
        "Loquercio, A., Kaufmann, E., Ranftl, R., Müller, M., Koltun, V., & Scaramuzza, D. (2021). Learning high-speed flight in the wild. Science Robotics, 6(59), eabg5810.",
        "Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. In Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI) (pp. 674-679).",
        "Macenski, S., Foote, T., Gerkey, B., Lalancette, C., & Woodall, W. (2022). Robot Operating System 2: Design, architecture, and uses in the wild. Science Robotics, 7(66), eabm6074.",
        "Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). ORB-SLAM: A versatile and accurate monocular SLAM system. IEEE Transactions on Robotics, 31(5), 1147-1163.",
        "Nistér, D. (2004). An efficient solution to the five-point relative pose problem. IEEE Transactions on Pattern Analysis and Machine Intelligence, 26(6), 756-770.",
        "Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A robust and versatile monocular visual-inertial state estimator. IEEE Transactions on Robotics, 34(4), 1004-1020.",
        "Ran, L., Zhang, Y., Zhang, Q., & Yang, T. (2017). Convolutional neural network-based visual odometry failure detection for mobile robots. International Journal of Advanced Robotic Systems, 14(6), 1-12.",
        "Scaramuzza, D., & Fraundorfer, F. (2011). Visual odometry: Part I: The first 30 years and tutorial. IEEE Robotics & Automation Magazine, 18(4), 80-92.",
        "Shi, J., & Tomasi, C. (1994). Good features to track. In 1994 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 593-600). IEEE.",
        "Wiens, J., Saria, S., Sendak, M., Ghassemi, M., Liu, V. X., Doshi-Velez, F., ... & Goldenberg, A. (2019). Do no harm: A roadmap for responsible machine learning for healthcare. Nature Medicine, 25(9), 1337-1340."
    ]
    for ref in references:
        p = builder.doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.10
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.left_indent = Inches(0.25)
        r = p.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.5)

    # =========================================================================
    # APPENDICES
    # =========================================================================
    builder.add_heading1("Appendix A: Exact Mathematical Metric Formulations")
    builder.add_paragraph(
        "Let y_i ∈ {0, 1} denote the binary ground-truth label and p_i = P(y_i = 1 | x_i) ∈ [0, 1] denote the model predicted probability "
        "for sample i ∈ {1, ..., N}. The Area Under the Receiver Operating Characteristic (AUROC) and Area Under the Precision-Recall Curve "
        "(AUPRC) are evaluated non-parametrically across all unique decision thresholds θ ∈ [0, 1]:"
    )
    builder.add_equation("TPR(θ) = [ ∑ y_i · 𝟙(p_i ≥ θ) ] / [ ∑ y_i ]")
    builder.add_equation("FPR(θ) = [ ∑ (1 - y_i) · 𝟙(p_i ≥ θ) ] / [ ∑ (1 - y_i) ]")
    builder.add_equation("Precision(θ) = TP(θ) / [ TP(θ) + FP(θ) ]")
    builder.add_equation("AUPRC = ∑_{k=1}^M [ Recall_k - Recall_{k-1} ] · Precision_k")
    builder.add_paragraph(
        "For event-level evaluation, let E = { (t_start^(j), t_end^(j)) }_{j=1}^J denote the set of contiguous failure episodes. "
        "The pre-onset anticipation window of width W is defined as W_j = [t_start^(j) - W, t_start^(j) - 1]. "
        "The rising-edge early warning indicator and any-active indicator are defined as:"
    )
    builder.add_equation("E_rising^(j) = 𝟙( ∃ t ∈ W_j :  𝟙(p_t ≥ θ) - 𝟙(p_{t-1} ≥ θ) = 1 )")
    builder.add_equation("E_active^(j) = 𝟙( ∃ t ∈ W_j :  p_t ≥ θ )")

    builder.add_page_break()
    builder.add_heading1("Appendix B: Per-Flight Breakdown Across All 14 Test Flights", full_width=True)
    builder.add_paragraph(
        "Table 8 provides the complete flight-by-flight performance breakdown across all 14 held-out test flights under strict clean evaluation "
        "(N = 9,758 frames), comparing the frozen model against retrained SmallMLP V0.",
        full_width=True
    )

    # Table 8: Per-flight breakdown
    df_per_flight = pd.read_csv(REPO_ROOT / "results" / "audit_final" / "per_flight_strict_test.csv")
    t8_headers = ["Flight", "Cell", "Frames", "Pos", "Rate", "AUROC (Fz)", "AUPRC (Fz)", "F1 (Fz)", "AUROC (V0)", "AUPRC (V0)", "F1 (V0)"]
    t8_data = []
    for _, row in df_per_flight.iterrows():
        fl_short = str(row["flight"]).replace("sweep_", "")
        t8_data.append([
            fl_short, str(row["family"]), str(row["total_frames"]), str(row["positive_frames"]),
            f"{float(row['positive_rate'])*100:.1f}%", f"{float(row['frozen_auroc']):.4f}",
            f"{float(row['frozen_auprc']):.4f}", f"{float(row['frozen_f1']):.4f}",
            f"{float(row['v0_auroc']):.4f}", f"{float(row['v0_auprc']):.4f}", f"{float(row['v0_f1']):.4f}"
        ])
    builder.add_table_with_caption(
        "Flight-by-Flight Strict Test Performance Across All 14 Quarantined Test Flights. "
        "Evaluated on clean non-failed frames (F_t = 0) with ongoing failures purged.",
        t8_headers, t8_data, [0.90, 0.45, 0.50, 0.45, 0.55, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62]
    )

    builder.add_heading1("Appendix C: Forensic Discrepancy Ledger & Historical Audit Trail", full_width=True)
    builder.add_paragraph(
        "To ensure complete scientific transparency, Table 9 summarizes the eight key methodological and reporting discrepancies "
        "identified between the legacy preprint draft and the verified ground truth established during our forensic audit.",
        full_width=True
    )

    # Table 9: Discrepancy Ledger
    t9_headers = ["Discrepancy Issue", "Legacy Value / Claim", "Audited Ground Truth", "Underlying Cause", "Status"]
    t9_data = [
        ["Primary Test AUROC", "AUROC = 0.8046, AUPRC = 0.7260", "AUROC = 0.7665, AUPRC = 0.5982", "Contemporaneous lag-0 flow leakage on already-failed frames", "CORRECTED"],
        ["Causal Replay Frames", "10,481 frames evaluated", "10,411 frames evaluated", "NaN-skipping in pandas max() retained 5 tail frames across 14 flights", "CORRECTED"],
        ["Alarm Rate Reporting", "~38.7 alarms/min cited in prose", "107.17 edges/min, 41.16% duty cycle", "Ambiguity between 0->1 rising transitions and frame duty cycle", "CLARIFIED"],
        ["Event Warning at 30% Duty", "57.3% (V0), 58.9% (HGB) in prose", "62.91% (V0), 79.13% (HGB)", "Author mistakenly copied frame-level recalls into event section", "CORRECTED"],
        ["Anticipation Horizon", "Capped at 0.165 s in text", "Extends to 0.50s (W=15) and 1.0s (W=30)", "Confusion between K=5 label horizon and W event lookback", "CORRECTED"],
        ["Model Architecture Claim", "SmallMLP superior to classical", "HistGradientBoosting beats MLP", "Cursory baseline tuning in original draft; HGB achieves 0.7733 AUROC", "CORRECTED"],
        ["Failure Mechanism Framing", "Prolonged VO collapse / drift", "99.59% single-frame skips (33.3 ms)", "KLT point loss triggers immediate corner re-detection in minimal_vo.py", "CORRECTED"],
        ["HGB Event Lead Time", "0.50s med / 0.443s mean in audit vs 0.367s / 0.331s in table", "Both verified: 0.367s / 0.331s (rising edge) vs 0.50s / 0.443s (any active)", "Operational ambiguity between initial rising edge (0->1) vs pre-existing active alarm state", "RESOLVED"]
    ]
    builder.add_table_with_caption(
        "Forensic Reconciliation and Discrepancy Ledger. "
        "Documents each historical error, its root cause, and the definitive audited correction.",
        t9_headers, t9_data, [1.30, 1.35, 1.35, 1.90, 0.80]
    )

    # Save document
    docx_path = DOCS_DIR / "AEGIS_Preprint.docx"
    print(f"Saving Word document to {docx_path}...")
    builder.doc.save(str(docx_path))
    print(f"Document saved successfully ({docx_path.stat().st_size} bytes).")

    # Convert to PDF via LibreOffice
    print("Converting DOCX to PDF via LibreOffice headless...")
    cmd = ["/snap/bin/libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(DOCS_DIR), str(docx_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"LibreOffice exit code: {res.returncode}")
    print(f"LibreOffice stdout: {res.stdout.strip()}")
    if res.stderr:
        print(f"LibreOffice stderr: {res.stderr.strip()}")

    pdf_path = DOCS_DIR / "AEGIS_Preprint.pdf"
    if pdf_path.exists():
        print(f"PDF generated successfully at {pdf_path} ({pdf_path.stat().st_size} bytes).")
    else:
        print("ERROR: PDF was not generated!")


if __name__ == "__main__":
    build_manuscript()
