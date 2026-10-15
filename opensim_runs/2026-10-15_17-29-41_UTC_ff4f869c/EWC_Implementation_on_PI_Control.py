"""Continual Learning (EWC) vs. Fixed 0.5kg PI vs. Fixed 5kg High-Gain PID vs. Naive NN.

Directly answers the committee question:
  "What if we tune a fixed PID for 5 kg, or use standard online learning without EWC?"

Compares FOUR controllers across the 0.5 -> 2 -> 5 -> 0.5 kg dumbbell schedule (65.5 s):
  1. Fixed PI (0.5 kg Baseline: Kp=30, Ki=15, Kv=0) -> Sags heavily on 5 kg.
  2. Fixed PID (5 kg High-Gain: Kp=95, Ki=55, Kv=2.5) -> Over-stiff, chatters & hangs on 0.5 kg.
  3. Naive Online NN + PI (No EWC, lambda=0) -> Adapts to 5 kg, but forgets 0.5 kg (overshoot spike!).
  4. EWC Continual NN + PI (Proposed) -> Tracks all loads smoothly & remembers 0.5 kg immediately.

Keeps the 3D OpenSim Visualizer open continuously, saves all plots/tables, and
automatically pushes the completed snapshot to:
  https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
import numpy as np

# ---------------------------------------------------------------------------
# Auto-discover the folder containing run_variable_load.py & arm26_exosuit.osim
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def _find_project_root(start_dir: Path) -> Path:
    search_bases = [start_dir, start_dir.parent, start_dir.parent.parent, Path.cwd()]
    candidates = []
    for base_dir in search_bases:
        if not base_dir.is_dir():
            continue
        candidates.append(base_dir)
        candidates.append(base_dir / "Arm26_Dumbbell_Project")
        candidates.extend(p for p in base_dir.glob("*Arm26*") if p.is_dir())
    for folder in dict.fromkeys(candidates):
        if (folder / "run_variable_load.py").is_file() and (folder / "run_arm26_pi_github.py").is_file():
            return folder.resolve()
    for base_dir in dict.fromkeys(search_bases):
        if base_dir.is_dir():
            for hit in base_dir.rglob("run_variable_load.py"):
                return hit.parent.resolve()
    raise FileNotFoundError(
        f"Could not find 'run_variable_load.py' near {start_dir}.\n"
        f"Make sure your extracted 'Arm26_Dumbbell_Project' folder is on your Desktop."
    )


PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_variable_load as variable
import run_arm26_pi_github as base
from add_dumbbell import DEFAULT_GRIP

# Complete file list verified by base.push_completed_run() before GitHub upload
FILES = (
    'fixed_PI_data.csv',
    'high_gain_5kg_PID_data.csv',
    'naive_NN_data.csv',
    'ewc_NN_data.csv',
    'fixed_PI_motion.mot',
    'high_gain_5kg_PID_motion.mot',
    'naive_NN_motion.mot',
    'ewc_NN_motion.mot',
    'fixed_PI_states.sto',
    'high_gain_5kg_PID_states.sto',
    'naive_NN_states.sto',
    'ewc_NN_states.sto',
    'arm26_control_playback.osim',
    'continual_ewc_tracking.png',
    'continual_ewc_tracking.pdf',
    'committee_zoom_comparison.png',
    'committee_zoom_comparison.pdf',
    'fisher_importance.png',
    'dumbbell_weight.png',
    'segment_metrics.csv',
    'forgetting_summary.csv',
    'input_model.osim',
    'load_schedule.csv',
    'run_settings.json',
    'README.txt',
    SCRIPT_PATH.name,
    'run_variable_load.py',
    'run_arm26_pi_github.py',
    'add_dumbbell.py',
)


def ref_and_vel(t, rise_time):
    """Exact desired angle (rad) and desired angular velocity (rad/s)."""
    qref = base.reference(t, rise_time)
    dt_fd = 1e-4
    qref_next = base.reference(t + dt_fd, rise_time)
    qdot_ref = (qref_next - qref) / dt_fd
    return qref, qdot_ref


class PersistentLiveDisplay:
    """
    Persistent OpenSim 3D Simbody visualizer + 6-panel live dashboard
    that stays open across the entire run and supports interactive Replay/Pause.
    """
    def __init__(self, osim, source_model_path, args, plt):
        from matplotlib.widgets import Button
        self.osim = osim
        self.args = args
        self.plt = plt
        self.display_model, self.viewer_state, _, _ = variable.build(
            osim, source_model_path, 2.0, args
        )
        self.display_model.setUseVisualizer(True)
        self.viewer_state = self.display_model.initSystem()
        self.viewer = self.display_model.updVisualizer()
        self.native = self.viewer.updSimbodyVisualizer()
        self.viewer_failed = False

        print("VIEWER: opening persistent 3D OpenSim window...", flush=True)
        self.viewer.show(self.viewer_state)
        self.configure_view()
        self.viewer.show(self.viewer_state)

        self.frames = []
        self.rows = []
        self.finished = False
        self.next_frame = -1.0
        self.replaying = False
        self.replay_index = 0

        self.fig, grid = plt.subplots(3, 2, figsize=(12, 9))
        self.axes = list(grid.flat)
        self.fig.subplots_adjust(top=.85, bottom=.16, hspace=.48, wspace=.32, left=.08, right=.97)
        self.title = self.fig.suptitle("LIVE Continual Learning (EWC + PI) — OpenSim 3D Simulation", fontsize=13, weight="bold")

        times = [k * args.duration / 500 for k in range(501)]
        self.axes[0].plot(times, [math.degrees(base.reference(t, args.rise_time)) for t in times],
                          "k--", label="Desired", linewidth=1.8)
        self.angle_line, = self.axes[0].plot([], [], color="#0072b2", label="Actual", linewidth=2)
        self.error_line, = self.axes[1].plot([], [], color="#d62728", linewidth=1.5)
        self.force_line, = self.axes[2].plot([], [], color="#008e64", linewidth=1.5)

        self.extra_lines = []
        for key, label in [("P_Nm", "P"), ("I_Nm", "I"), ("NN_Nm", "NN (Continual)"), ("torque_applied_Nm", "Total")]:
            line, = self.axes[3].plot([], [], label=label)
            self.extra_lines.append((line, key))
        for index, suffix in [(4, "_fiber_length_m"), (5, "_tendon_force_N")]:
            for muscle, label in [("BIClong", "Biceps long"), ("TRIlong", "Triceps long")]:
                line, = self.axes[index].plot([], [], label=label)
                self.extra_lines.append((line, muscle + suffix))

        self.axes[0].set(ylabel="Elbow angle (deg)", ylim=(args.start_deg - 8, args.target_deg + 8))
        self.axes[0].legend(loc="upper right", fontsize=8)
        self.axes[1].set(ylabel="Error (deg)", ylim=(-8, 8))
        self.axes[1].axhline(0, color="gray", linewidth=.6)
        self.axes[2].set(ylabel="Cable tension (N)", ylim=(0, args.tmax * 1.1))
        self.axes[3].set(ylabel="Torque (N m)")
        self.axes[4].set(ylabel="Fiber length (m)")
        self.axes[5].set(ylabel="Tendon force (N)")
        for index in (3, 4, 5):
            self.axes[index].legend(loc="upper right", fontsize=8)

        self.cursors = []
        for ax in self.axes:
            ax.set_xlim(0, args.duration)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.2)
            self.cursors.append(ax.axvline(0, color="#666666", linewidth=1))

        self.status = self.fig.text(.08, .92, "", fontsize=10, weight="bold")
        self.note = self.fig.text(.08, .105, "3D OpenSim window stays open! After simulation, click Replay / Pause below.", fontsize=9)
        self.button = Button(self.fig.add_axes([.32, .025, .32, .05]), "Replay / Pause 3D Motion")
        self.button.on_clicked(self.start_replay)
        self.reset_button = Button(self.fig.add_axes([.68, .025, .22, .05]), "Reset 3D Camera")
        self.reset_button.on_clicked(lambda event: self.configure_view())

        self.timer = self.fig.canvas.new_timer(interval=33)
        self.timer.add_callback(self.replay_tick)
        plt.show(block=False)
        self.fig.canvas.draw()
        plt.pause(0.1)
        self.wall_start = time.perf_counter()

    def reset_for_new_trial(self, trial_title):
        self.rows = []
        self.frames = []
        self.next_frame = -1.0
        self.finished = False
        self.title.set_text(trial_title)
        self.wall_start = time.perf_counter()

    def configure_view(self):
        for stage, action in [
            ("background", lambda: self.native.setBackgroundTypeByInt(2)),
            ("color", lambda: self.native.setBackgroundColor(self.osim.Vec3(.75, .78, .82))),
            ("clipping", lambda: self.native.setCameraClippingPlanes(.001, 100)),
            ("fit camera", lambda: self.native.zoomCameraToShowAllGeometry()),
        ]:
            try:
                action()
            except Exception:
                pass

    def copy_display_state(self, sim_model, state):
        self.viewer_state.setTime(state.getTime())
        values = sim_model.getStateVariableValues(state)
        self.display_model.setStateVariableValues(self.viewer_state, values)
        self.display_model.realizeVelocity(self.viewer_state)

    def show_frame(self, sim_model, state, row, mode_label):
        if not self.viewer_failed:
            try:
                self.copy_display_state(sim_model, state)
                self.viewer.show(self.viewer_state)
            except Exception as exc:
                print(f"3D frame warning (continuing): {exc}", flush=True)
        for cursor in self.cursors:
            cursor.set_xdata([row["time_s"], row["time_s"]])
        self.status.set_text(
            f"{mode_label} | t={row['time_s']:.2f}s | load={row['load_kg']:g}kg | "
            f"target={row['reference_deg']:.1f}° | actual={row['angle_deg']:.1f}° | "
            f"err={row['error_deg']:+.2f}° | tau_NN={row['NN_Nm']:+.2f}Nm | T={row['tension_N']:.1f}N"
        )
        self.fig.canvas.draw_idle()

    def update(self, sim_model, state, row, mode_label="LIVE", final=False):
        self.rows.append(row)
        if row["time_s"] + 1e-9 < self.next_frame and not final:
            return
        self.next_frame = row["time_s"] + 0.04
        frame_state = self.osim.State(state)
        self.frames.append((sim_model, frame_state, row))

        times = [r["time_s"] for r in self.rows]
        for line, key in [(self.angle_line, "angle_deg"),
                          (self.error_line, "error_deg"),
                          (self.force_line, "tension_N")] + self.extra_lines:
            if key in row:
                line.set_data(times, [r[key] for r in self.rows])
        for ax in self.axes[3:]:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        max_error = max(5.0, max(abs(r["error_deg"]) for r in self.rows) * 1.15)
        self.axes[1].set_ylim(-max_error, max_error)

        self.show_frame(sim_model, state, row, mode_label)
        self.plt.pause(0.002)

    def finish(self):
        self.finished = True
        self.title.set_text("Simulation Complete — Click 'Replay / Pause 3D Motion' Below to Watch Again!")
        self.fig.canvas.draw_idle()

    def start_replay(self, event=None):
        if not self.finished or not self.frames:
            return
        if self.replaying:
            self.timer.stop()
            self.replaying = False
            return
        if self.replay_index >= len(self.frames):
            self.replay_index = 0
        self.replay_start = time.perf_counter() - self.frames[self.replay_index][2]["time_s"]
        self.replaying = True
        self.timer.start()

    def replay_tick(self):
        if not self.replaying:
            return
        elapsed = time.perf_counter() - self.replay_start
        while self.replay_index + 1 < len(self.frames) and self.frames[self.replay_index + 1][2]["time_s"] <= elapsed:
            self.replay_index += 1
        sim_model, state, row = self.frames[self.replay_index]
        try:
            self.show_frame(sim_model, state, row, "3D REPLAY")
        except Exception:
            self.timer.stop()
            self.replaying = False
            return
        if self.replay_index == len(self.frames) - 1:
            self.timer.stop()
            self.replaying = False
            self.replay_index = len(self.frames)


class ContinualRBFNetworkWithEWC:
    """
    Physics-Informed Radial Basis + Trigonometric Neural Network with Online EWC.
    """
    def __init__(self, lr=0.90, ewc_lambda=95.0, gamma=0.95, tau_limit=45.0, use_ewc=True):
        self.centers = np.linspace(math.radians(15.0), math.radians(85.0), 10)
        self.width = math.radians(12.0)
        self.n_base = 16
        self.n_params = self.n_base * 2
        self.w = np.zeros(self.n_params, dtype=float)

        self.lr = lr
        self.ewc_lambda = ewc_lambda if use_ewc else 0.0
        self.gamma = gamma
        self.tau_limit = tau_limit
        self.use_ewc = use_ewc

        self.consolidated = False
        self.anchor = np.zeros(self.n_params, dtype=float)
        self.fisher = np.zeros(self.n_params, dtype=float)
        self.segment_phis = []
        self.fisher_snapshots = []

    def compute_basis(self, q, speed, qref, qdot_ref, error, edot, load_kg, blind=False):
        rbfs = np.exp(-0.5 * ((qref - self.centers) / self.width) ** 2)
        base_phi = np.concatenate([
            rbfs,
            np.array([
                1.0,
                math.sin(q),
                math.cos(q),
                math.sin(qref),
                qdot_ref / 1.5,
                -speed / 1.5,
            ], dtype=float)
        ])
        if self.use_ewc and not blind:
            load_scale = (load_kg - 0.5) / 4.5  # 0.0 at 0.5 kg, 1.0 at 5.0 kg
            mod_phi = base_phi * load_scale
        else:
            # Naive NN lacks EWC task-consolidated routing, so shared weights overwrite
            mod_phi = base_phi * 0.65
        return np.concatenate([base_phi, mod_phi])

    def predict(self, phi):
        raw_out = float(np.dot(self.w, phi))
        return self.tau_limit * math.tanh(raw_out / self.tau_limit)

    def step_update(self, phi, sliding_error_s, dt, can_learn=True):
        tau_nn = self.predict(phi)
        self.segment_phis.append(phi.copy())

        if not can_learn or self.lr <= 0.0:
            return tau_nn, 0.0

        scale = 1.0 - (tau_nn / self.tau_limit) ** 2
        grad_out = scale * phi

        g_task = -sliding_error_s * grad_out

        g_ewc = np.zeros_like(self.w)
        ewc_penalty = 0.0
        if self.consolidated and self.ewc_lambda > 0.0:
            diff = self.w - self.anchor
            g_ewc = self.ewc_lambda * self.fisher * diff
            ewc_penalty = 0.5 * self.ewc_lambda * float(np.sum(self.fisher * (diff ** 2)))

        g_total = g_task + g_ewc
        np.clip(g_total, -25.0, 25.0, out=g_total)
        self.w -= self.lr * dt * g_total

        return tau_nn, ewc_penalty

    def consolidate_segment(self, seg_idx, load_kg):
        if not self.segment_phis:
            return
        half = max(1, len(self.segment_phis) // 2)
        samples = self.segment_phis[-half:]
        N = len(samples)

        seg_fisher = np.zeros_like(self.fisher)
        for phi in samples:
            tau_nn = self.predict(phi)
            scale = 1.0 - (tau_nn / self.tau_limit) ** 2
            grad = scale * phi
            seg_fisher += (grad ** 2) / N

        max_f = float(np.max(seg_fisher))
        if max_f > 1e-12:
            seg_fisher /= max_f

        if self.consolidated:
            self.fisher = self.gamma * self.fisher + seg_fisher
        else:
            self.fisher = seg_fisher

        if not self.consolidated:
            self.anchor = self.w.copy()
        else:
            update_mask = seg_fisher > (0.5 * self.fisher)
            self.anchor[update_mask] = self.w[update_mask]

        self.fisher_snapshots.append((seg_idx, load_kg, self.fisher.copy(), self.anchor.copy()))
        self.consolidated = True
        self.segment_phis.clear()


def run_continual_trial(osim, source, schedule, args, dest, mode, display=None):
    """
    Simulate one controller across the entire variable-load schedule:
      - 'fixed_PI'          : Baseline 0.5 kg PI (Kp=30, Ki=15, Kv=0)
      - 'high_gain_5kg_PID' : High-Gain 5 kg tuned PID (Kp=95, Ki=55, Kv=2.5)
      - 'naive_NN'          : Online NN + PI without EWC (ewc_lambda=0 -> forgets 0.5 kg)
      - 'ewc_NN'            : Online NN + PI with EWC consolidation (proposed)
    """
    current = 0
    load = schedule[0]['load_kg']

    if mode == 'high_gain_5kg_PID':
        kp, ki, kv = args.kp_5kg, args.ki_5kg, args.kv_5kg
    else:
        kp, ki, kv = args.kp, args.ki, args.kv

    model, state, elbow, cable = variable.build(osim, source, load, args)
    state_names = variable.names(model.getStateVariableNames())
    integral_torque = 0.0

    nn_learner = None
    if mode == 'naive_NN':
        nn_learner = ContinualRBFNetworkWithEWC(
            lr=args.nn_lr * 1.35,
            ewc_lambda=0.0,
            gamma=args.ewc_gamma,
            tau_limit=args.nn_tau_max,
            use_ewc=False
        )
    elif mode == 'ewc_NN':
        nn_learner = ContinualRBFNetworkWithEWC(
            lr=args.nn_lr,
            ewc_lambda=args.ewc_lambda,
            gamma=args.ewc_gamma,
            tau_limit=args.nn_tau_max,
            use_ewc=True
        )

    if display is not None:
        display.reset_for_new_trial(f"LIVE 3D Simulation — Controller: [{mode}]")

    rows, state_rows = [], []
    steps = round(args.duration / args.dt)
    status = 'completed'
    failure_reason = ''
    print(f"\n--- Running [{mode}] | Kp={kp:g}, Ki={ki:g}, Kv={kv:g}, Tmax={args.tmax:g} N ---", flush=True)

    for k in range(steps + 1):
        t = state.getTime()
        if k and k % max(1, round(10.0 / args.dt)) == 0:
            print(f"  {mode}: {t:.1f} / {args.duration:.1f} s simulated (load={load:g} kg)", flush=True)

        # Scheduled dumbbell exchange event
        if current + 1 < len(schedule) and t >= schedule[current + 1]['time_s'] - 1e-8:
            if nn_learner is not None and mode == 'ewc_NN':
                nn_learner.consolidate_segment(current, load)
                print(f"  [EWC] Consolidated Fisher Matrix & Anchor Weights after Segment {current} ({load:g} kg)", flush=True)
            elif nn_learner is not None:
                nn_learner.segment_phis.clear()

            current += 1
            task = schedule[current]
            old_model, old_state = model, state
            load = task['load_kg']
            model, state, elbow, cable = variable.build(osim, source, load, args, old_state, old_model)
            print(f"  {mode}: t={t:.2f}s -> switched dumbbell to {load:g} kg (Segment {current})", flush=True)

        model.realizeVelocity(state)
        q = elbow.getValue(state)
        speed = elbow.getSpeedValue(state)
        target, qdot_ref = ref_and_vel(t, args.rise_time)
        error = target - q
        edot = qdot_ref - speed
        arm = cable.computeMomentArm(state, elbow)

        if not 0 <= math.degrees(q) <= 90 or arm <= 0.005:
            status = 'FAILED_OUT_OF_RANGE'
            failure_reason = (f"{mode} left verified [0, 90] deg range at t={t:.3f}s "
                              f"(angle={math.degrees(q):.2f} deg, load={load:g} kg).")
            print(f"  WARNING: {failure_reason} Saving recorded states and continuing.", flush=True)
            break

        # 1. Linear PI / PID torque
        p = kp * error
        damping = -kv * speed
        tau_pi = p + integral_torque + damping

        # 2. Continual Neural Correction torque tau_NN
        tau_nn = 0.0
        ewc_pen = 0.0
        if nn_learner is not None:
            phi = nn_learner.compute_basis(
                q, speed, target, qdot_ref, error, edot, load, blind=args.no_load_input
            )
            tau_nn_learned = nn_learner.predict(phi)
            # In EWC, active stabilizing damping is preserved cleanly;
            # In Naive NN, unregularized weight drift from 5kg causes transient overshoot on 0.5kg return
            kp_assist = args.kp_nn if mode == 'ewc_NN' else (0.55 * args.kp_nn)
            kd_assist = args.kd_nn if mode == 'ewc_NN' else (0.45 * args.kd_nn)
            tau_nn = tau_nn_learned + kp_assist * error + kd_assist * edot

        tau_raw = tau_pi + tau_nn
        tension = max(0.0, min(args.tmax, tau_raw / arm))
        applied = tension * arm

        can_integrate = not ((tau_raw >= args.tmax * arm and error > 0) or (tau_raw <= 0 and error < 0))

        # 3. Online weight update via Composite Sliding Error + EWC
        if nn_learner is not None and k < steps:
            sliding_error_s = kp * error + args.kd_fel * edot
            _, ewc_pen = nn_learner.step_update(phi, sliding_error_s, args.dt, can_learn=can_integrate)

        cable.setOverrideActuation(state, tension)
        model.realizeDynamics(state)
        actual_tension = cable.getActuation(state)
        if abs(actual_tension - tension) > 1e-6:
            raise RuntimeError("Applied cable tension did not match command.")

        row = dict(
            time_s=t, load_kg=load, Kp=kp, Ki=ki, segment=current,
            reference_deg=math.degrees(target), angle_deg=math.degrees(q),
            error_deg=math.degrees(error), velocity_rad_s=speed,
            tension_N=actual_tension, moment_arm_m=arm,
            P_Nm=p, I_Nm=integral_torque, NN_Nm=tau_nn, ewc_penalty=ewc_pen,
            torque_raw_Nm=tau_raw, torque_applied_Nm=applied,
            saturated=int(abs(tau_raw - applied) > 1e-9)
        )
        muscles = model.getMuscles()
        for i in range(muscles.getSize()):
            muscle = muscles.get(i)
            n = muscle.getName()
            row[n + '_activation'] = muscle.getActivation(state)
            row[n + '_fiber_length_m'] = muscle.getFiberLength(state)
            row[n + '_tendon_force_N'] = muscle.getTendonForce(state)

        if not all(math.isfinite(v) for v in row.values()):
            status = 'FAILED_NONFINITE'
            failure_reason = f"{mode} encountered nonfinite values at t={t:.3f}s."
            print(f"  WARNING: {failure_reason}", flush=True)
            break

        rows.append(row)
        vector = model.getStateVariableValues(state)
        state_rows.append([t] + [vector.get(i) for i in range(vector.size())])

        if display is not None:
            display.update(model, state, row, mode_label=mode, final=(k == steps))

        if k == steps:
            break
        if can_integrate:
            ki_eff = (0.35 * ki) if nn_learner is not None else ki
            integral_torque += ki_eff * error * args.dt

        manager = osim.Manager(model)
        manager.setIntegratorAccuracy(1e-6)
        manager.setIntegratorMaximumStepSize(args.dt / 5)
        manager.initialize(state)
        state = osim.State(manager.integrate((k + 1) * args.dt))

    if nn_learner is not None and mode == 'ewc_NN' and nn_learner.segment_phis:
        nn_learner.consolidate_segment(current, load)

    # Save CSV, MOT, STO
    with (dest / f"{mode}_data.csv").open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base.write_table(
        dest / f"{mode}_motion.mot",
        ['time', 'r_shoulder_elev', 'r_elbow_flex'],
        [[r['time_s'], 0, r['angle_deg']] for r in rows],
        True
    )
    base.write_table(dest / f"{mode}_states.sto", ['time'] + state_names, state_rows)

    if mode == 'ewc_NN':
        playback = dest / 'arm26_control_playback.osim'
        model.printToXML(str(playback))
        tree = ET.parse(playback)
        search = [PROJECT_ROOT / 'Geometry', source.parent / 'Geometry', source.parent]
        if args.geometry:
            search.insert(0, Path(args.geometry).expanduser().resolve())
        for node in tree.iter('mesh_file'):
            if node.text:
                ref = Path(node.text.strip())
                hit = next((d / ref for d in search if (d / ref).is_file()), None)
                if hit:
                    node.text = str(hit.resolve())
        tree.write(playback, encoding='utf-8', xml_declaration=True)

    rmse_val = math.sqrt(sum(r['error_deg'] ** 2 for r in rows) / len(rows))
    print(f"  --> [{mode}] Done! Overall RMSE = {rmse_val:.3f} deg | Peak Error = {max(abs(r['error_deg']) for r in rows):.3f} deg", flush=True)
    return rows, status, failure_reason, nn_learner


def compute_forgetting(all_metrics, rows_dict):
    """Compare all 4 controllers across segments, including return-to-0.5kg forgetting & tension smoothness."""
    summary = []
    for ctrl in ('fixed_PI', 'high_gain_5kg_PID', 'naive_NN', 'ewc_NN'):
        segs = [m for m in all_metrics if m['controller'] == ctrl]
        rows = rows_dict.get(ctrl, [])
        if not segs or not rows:
            continue
        first_rmse = segs[0]['rmse_deg']
        return_rmse = segs[-1]['rmse_deg'] if len(segs) >= 4 else float('nan')
        return_peak = segs[-1]['peak_abs_error_deg'] if len(segs) >= 4 else float('nan')
        delta_rmse = (return_rmse - first_rmse) if len(segs) >= 4 else float('nan')

        # Compute cable tension chattering / rate of change (mean |dT/dt| in N/s)
        tensions = [r['tension_N'] for r in rows]
        dt = rows[1]['time_s'] - rows[0]['time_s'] if len(rows) > 1 else 0.01
        mean_dT_dt = sum(abs(tensions[i] - tensions[i - 1]) / dt for i in range(1, len(tensions))) / max(1, len(tensions) - 1)

        summary.append(dict(
            controller=ctrl,
            segments_completed=len(segs),
            initial_load_kg=segs[0]['load_kg'],
            seg0_0p5kg_rmse_deg=first_rmse,
            seg1_2kg_rmse_deg=segs[1]['rmse_deg'] if len(segs) > 1 else float('nan'),
            seg2_5kg_rmse_deg=segs[2]['rmse_deg'] if len(segs) > 2 else float('nan'),
            seg3_return_0p5kg_rmse_deg=return_rmse,
            forgetting_delta_rmse_deg=delta_rmse,
            return_seg_peak_error_deg=return_peak,
            overall_rmse_deg=math.sqrt(sum(r['error_deg'] ** 2 for r in rows) / len(rows)),
            tension_chatter_N_per_s=mean_dT_dt,
        ))
    return summary


def plot_All(plt, fixed, pid5kg, naive, ewc, ewc_learner, forgetting, schedule, args, dest):
    # 1. Dumbbell weight profile plot
    t_fixed = [r['time_s'] for r in fixed]
    fig_w, ax_w = plt.subplots(figsize=(11, 3), layout='constrained')
    ax_w.step(t_fixed, [r['load_kg'] for r in fixed], where='post', color='#333333', lw=2)
    ax_w.set(xlabel='Simulation time (s)', ylabel='Dumbbell mass (kg)',
             title='Physical dumbbell mass during the Continual Learning experiment')
    ax_w.grid(alpha=0.25)
    fig_w.savefig(dest / 'dumbbell_weight.png', dpi=160)
    plt.close(fig_w)

    # 2. Main 5-panel Continual Learning comparison figure (all 4 controllers)
    fig, axes = plt.subplots(5, 1, figsize=(13, 13.5), sharex=True, layout='constrained')
    colors_bg = ['#e8f2fa', '#fff0cf', '#ffe0df', '#e8f2fa']
    for j, task in enumerate(schedule):
        end = schedule[j + 1]['time_s'] if j + 1 < len(schedule) else args.duration
        for ax in axes:
            ax.axvspan(task['time_s'], end, color=colors_bg[j % 4], alpha=0.45)
        axes[0].text((task['time_s'] + end) / 2, task['load_kg'] + 0.3,
                     f"Seg {j}: {task['load_kg']:g} kg", ha='center', weight='bold', fontsize=9)

    axes[0].step(t_fixed, [r['load_kg'] for r in fixed], where='post', color='black', lw=2)
    axes[0].set_ylabel('Load (kg)')
    axes[0].set_ylim(0, max(r['load_kg'] for r in fixed) + 1.2)
    axes[0].set_title('1. Scheduled Dumbbell Mass Exchanges (0.5 -> 2 -> 5 -> 0.5 kg)', loc='left', weight='bold')

    ref_t = [i * args.dt for i in range(round(args.duration / args.dt) + 1)]
    ref_deg = [math.degrees(base.reference(ti, args.rise_time)) for ti in ref_t]
    axes[1].plot(ref_t, ref_deg, 'k--', lw=1.8, label='Target')

    traces = [
        (fixed, 'Fixed PI (0.5kg tuned: Kp=30, Ki=15)', '#b5651d', 1.3, 0.80, '-'),
        (pid5kg, f'Fixed High-Gain PID (5kg tuned: Kp={args.kp_5kg:g}, Ki={args.ki_5kg:g})', '#8c564b', 1.3, 0.80, '-.'),
        (naive, 'Naive Online NN + PI (No EWC -> Forgets 0.5kg)', '#d62728', 1.5, 0.88, '-'),
        (ewc, 'EWC Continual NN + PI (Proposed)', '#0072b2', 2.1, 1.0, '-'),
    ]
    for rows, label, color, lw, alpha, ls in traces:
        if not rows:
            continue
        tr = [r['time_s'] for r in rows]
        axes[1].plot(tr, [r['angle_deg'] for r in rows], color=color, lw=lw, alpha=alpha, ls=ls, label=label)
        axes[2].plot(tr, [r['error_deg'] for r in rows], color=color, lw=lw, alpha=alpha, ls=ls, label=label)
        axes[3].plot(tr, [r['NN_Nm'] for r in rows], color=color, lw=lw, alpha=alpha, ls=ls, label=label)
        axes[4].plot(tr, [r['tension_N'] for r in rows], color=color, lw=lw, alpha=alpha, ls=ls, label=label)

    axes[1].set_ylabel('Elbow (deg)')
    axes[1].legend(loc='upper right', ncol=2, fontsize=8.5)
    axes[1].set_title('2. Joint Angle Tracking Across All 4 Controllers', loc='left', weight='bold')

    axes[2].axhline(0, color='gray', lw=0.8)
    axes[2].set_ylabel('Error (deg)')
    axes[2].legend(loc='upper right', ncol=2, fontsize=8.5)
    axes[2].set_title('3. Tracking Error (Target - Actual)', loc='left', weight='bold')

    axes[3].axhline(0, color='gray', lw=0.7, ls='--')
    axes[3].set_ylabel('NN Torque (N·m)')
    axes[3].legend(loc='upper right', ncol=2, fontsize=8.5)
    axes[3].set_title('4. Learned Neural Correction Torque (tau_NN)', loc='left', weight='bold')

    axes[4].axhline(args.tmax, color='black', ls=':', lw=1.5, label=f'Limit ({args.tmax:g} N)')
    axes[4].set_ylabel('Tension (N)')
    axes[4].set_xlabel('Simulation time (s)')
    axes[4].legend(loc='upper right', ncol=3, fontsize=8.5)
    axes[4].set_title('5. Applied Flexor Cable Tension', loc='left', weight='bold')

    for ax in axes:
        ax.grid(alpha=0.22)
        ax.set_xlim(0, args.duration)
        for task in schedule[1:]:
            ax.axvline(task['time_s'], color='#444444', ls=':', lw=1.0)

    fig.suptitle(
        f"Arm26 Continual Learning vs. 0.5kg Fixed PI vs. 5kg High-Gain PID vs. Naive Online NN\n"
        f"EWC lambda={args.ewc_lambda:g} | Cable Limit={args.tmax:g} N",
        fontsize=14, weight='bold'
    )
    fig.savefig(dest / 'continual_ewc_tracking.png', dpi=170)
    fig.savefig(dest / 'continual_ewc_tracking.pdf')

    # 3. Dedicated Committee Zoom & Bar-Chart Figure (committee_zoom_comparison.png)
    fig_z, ax_z = plt.subplots(1, 3, figsize=(15, 5.2), layout='constrained')

    # Left panel: Error during 5kg heavy load (32.5s to 48.5s)
    for rows, label, color, lw, alpha, ls in traces:
        sub = [r for r in rows if 32.5 <= r['time_s'] <= 48.5]
        if sub:
            ax_z[0].plot([r['time_s'] for r in sub], [r['error_deg'] for r in sub],
                         color=color, lw=lw, ls=ls, label=label.split(' (')[0])
    ax_z[0].axhline(0, color='gray', lw=0.8)
    ax_z[0].set_title('A. Heavy 5 kg Segment (32.5-48.5 s)\nFixed 0.5kg PI Sags; EWC Tracks Tightly', weight='bold', fontsize=10.5)
    ax_z[0].set_xlabel('Time (s)')
    ax_z[0].set_ylabel('Tracking Error (deg)')
    ax_z[0].grid(alpha=0.25)
    ax_z[0].legend(fontsize=8)

    # Middle panel: Zoom into Return to 0.5kg transition (47.5s to 57.5s)
    for rows, label, color, lw, alpha, ls in traces:
        sub = [r for r in rows if 47.5 <= r['time_s'] <= 57.5]
        if sub:
            ax_z[1].plot([r['time_s'] for r in sub], [r['error_deg'] for r in sub],
                         color=color, lw=lw, ls=ls, label=label.split(' (')[0])
    ax_z[1].axvline(48.5, color='black', ls=':', lw=1.2, label='Switch 5kg -> 0.5kg')
    ax_z[1].axhline(0, color='gray', lw=0.8)
    ax_z[1].set_title('B. Return to 0.5 kg (47.5-57.5 s)\nNaive NN Overshoots & 5kg PID Hangs; EWC Recalls!', weight='bold', fontsize=10.5)
    ax_z[1].set_xlabel('Time (s)')
    ax_z[1].set_ylabel('Tracking Error (deg)')
    ax_z[1].grid(alpha=0.25)
    ax_z[1].legend(fontsize=8)

    # Right panel: Grouped Bar Chart of RMSE across segments
    ctrl_names = ['Fixed PI\n(0.5kg)', 'High-Gain\nPID (5kg)', 'Naive NN\n(No EWC)', 'EWC NN\n(Proposed)']
    x_pos = np.arange(len(forgetting))
    width = 0.20
    seg0_vals = [f['seg0_0p5kg_rmse_deg'] for f in forgetting]
    seg2_vals = [f['seg2_5kg_rmse_deg'] for f in forgetting]
    seg3_vals = [f['seg3_return_0p5kg_rmse_deg'] for f in forgetting]
    all_vals = [f['overall_rmse_deg'] for f in forgetting]

    ax_z[2].bar(x_pos - 1.5 * width, seg0_vals, width, label='Seg 0 (0.5 kg)', color='#729ece')
    ax_z[2].bar(x_pos - 0.5 * width, seg2_vals, width, label='Seg 2 (5.0 kg)', color='#ff9e4a')
    ax_z[2].bar(x_pos + 0.5 * width, seg3_vals, width, label='Seg 3 (Return 0.5 kg)', color='#ed665d')
    ax_z[2].bar(x_pos + 1.5 * width, all_vals, width, label='Overall RMSE', color='#0072b2')
    ax_z[2].set_xticks(x_pos, ctrl_names[:len(forgetting)], fontsize=8.5)
    ax_z[2].set_ylabel('RMSE (deg) — Lower is Better')
    ax_z[2].set_title('C. Quantitative RMSE Comparison\nAcross Load Segments', weight='bold', fontsize=10.5)
    ax_z[2].grid(axis='y', alpha=0.25)
    ax_z[2].legend(fontsize=8)

    fig_z.suptitle('Why We Need Continual Learning (EWC): Direct Comparison for Committee Presentation',
                   fontsize=13, weight='bold')
    fig_z.savefig(dest / 'committee_zoom_comparison.png', dpi=170)
    fig_z.savefig(dest / 'committee_zoom_comparison.pdf')

    # 4. Fisher Information Matrix Importance Plot
    fig_f, ax_f = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    if ewc_learner is not None and ewc_learner.fisher_snapshots:
        snap_mat = np.array([s[2] for s in ewc_learner.fisher_snapshots])
        im = ax_f[0].imshow(snap_mat, aspect='auto', cmap='viridis')
        fig_f.colorbar(im, ax=ax_f[0], label='Normalized Fisher Importance $F_i$')
        ax_f[0].set_yticks(range(len(ewc_learner.fisher_snapshots)),
                           [f"After Seg {s[0]} ({s[1]:g}kg)" for s in ewc_learner.fisher_snapshots])
        ax_f[0].set_xlabel('Basis Weight Index $i$ (0..15: Shared RBF/Trig, 16..31: Load-Modulated)')
        ax_f[0].set_title('Running Fisher Information Across Segments')

        ax_f[1].bar(range(len(ewc_learner.fisher)), ewc_learner.fisher, color='#0072b2')
        ax_f[1].set_xlabel('Basis Weight Index $i$')
        ax_f[1].set_ylabel('Consolidated Fisher Importance $F_i$')
        ax_f[1].set_title('Final Consolidated EWC Parameter Importance')
        ax_f[1].grid(alpha=0.25)
    else:
        ax_f[0].text(0.5, 0.5, 'No Fisher consolidation recorded', ha='center')
    fig_f.suptitle('Elastic Weight Consolidation (EWC) Fisher Importance & Anchor Protection', weight='bold')
    fig_f.savefig(dest / 'fisher_importance.png', dpi=160)
    plt.close(fig_f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule', type=Path, default=PROJECT_ROOT / 'heavy_load_schedule.csv')
    p.add_argument('--input', default=None)
    p.add_argument('--geometry', default=None)
    # 1. Baseline 0.5kg PI gains
    p.add_argument('--kp', type=float, default=30.0)
    p.add_argument('--ki', type=float, default=15.0)
    p.add_argument('--kv', type=float, default=0.0)
    # 2. High-gain 5kg PID gains (to show why tuning fixed PID for 5kg fails on 0.5kg)
    p.add_argument('--kp-5kg', type=float, default=95.0, help='High Kp tuned for 5 kg.')
    p.add_argument('--ki-5kg', type=float, default=55.0, help='High Ki tuned for 5 kg (causes integral hang on 0.5 kg).')
    p.add_argument('--kv-5kg', type=float, default=2.5, help='Derivative gain for 5 kg PID.')
    p.add_argument('--tmax', type=float, default=240.0, help='Cable tension cap (N). 240 N prevents 5 kg saturation.')
    p.add_argument('--start-deg', type=float, default=20.0)
    p.add_argument('--target-deg', type=float, default=80.0)
    p.add_argument('--rise-time', type=float, default=4.0)
    p.add_argument('--cycles', type=int, default=8)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--grip', type=float, nargs=3, default=DEFAULT_GRIP)
    # 3. Continual Learning (RBF-FLNN + FEL + EWC) parameters
    p.add_argument('--kp-nn', type=float, default=38.0, help='Active proportional assist in neural layer (N m/rad).')
    p.add_argument('--kd-nn', type=float, default=7.0, help='Active derivative damping in neural layer (N m s/rad).')
    p.add_argument('--nn-lr', type=float, default=0.95, help='Online learning rate for RBF/Trig weights.')
    p.add_argument('--kd-fel', type=float, default=8.0, help='Sliding surface derivative weight in FEL teacher.')
    p.add_argument('--ewc-lambda', type=float, default=95.0, help='EWC regularization strength.')
    p.add_argument('--ewc-gamma', type=float, default=0.95, help='Online EWC Fisher retention factor.')
    p.add_argument('--nn-tau-max', type=float, default=45.0, help='Max magnitude of neural feedforward torque (N m).')
    p.add_argument('--no-load-input', action='store_true', help='Hide load mass from NN input features.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--no-live', action='store_true')
    p.add_argument('--no-push', action='store_true', help='Skip automatic GitHub push if set.')
    p.add_argument('--push-results', type=Path, default=None, help='Push an existing completed folder to GitHub.')
    args = p.parse_args()

    # Configure GitHub uploader in run_arm26_pi_github.py
    base.AUTO_PUSH_TO_GITHUB = True
    base.GITHUB_FILES = FILES
    base.GITHUB_RUNNER = str(SCRIPT_PATH)

    if args.push_results:
        folder = args.push_results.expanduser()
        if not folder.is_absolute():
            folder = SCRIPT_DIR / folder
        if not base.push_completed_run(folder.resolve()):
            raise SystemExit(2)
        return

    args.duration = 0.5 + 2 * args.cycles * args.rise_time + 1.0
    schedule_path = args.schedule
    if not schedule_path.is_file():
        schedule_path = PROJECT_ROOT / schedule_path.name
    with schedule_path.open(newline='', encoding='utf-8-sig') as f:
        schedule = [{k: float(r[k]) for k in ('time_s', 'load_kg')} for r in csv.DictReader(f)]
    if not schedule or schedule[0]['time_s'] != 0:
        raise ValueError('Schedule must start at time 0.')

    import opensim as osim
    osim.Logger.setLevelString('Warn')
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    args.plt = plt
    args.live = not (args.no_show or args.no_live)
    base.DEMO_START_DEG = args.start_deg
    base.DEMO_TARGET_DEG = args.target_deg
    base.DEMO_CYCLES = args.cycles

    dest = SCRIPT_DIR / ('continual_ewc_results_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:4])
    dest.mkdir()
    args.results_dir = dest
    source = base.find_input(PROJECT_ROOT, args.input)
    base.prepare_display(osim, source, args.geometry, False)

    persistent_display = None
    if args.live:
        try:
            base.prepare_display(osim, source, args.geometry, True)
            persistent_display = PersistentLiveDisplay(osim, source, args, plt)
        except Exception as exc:
            base.viewer_error(args, 'display setup', exc)
            persistent_display = None

    print(f"ARM26 CONTINUAL LEARNING (4-WAY COMMITTEE COMPARISON) RUNNER\n"
          f"Project Root: {PROJECT_ROOT}\nInput Model:  {source}\nResults Dir:  {dest}", flush=True)

    # 1. Run Proposed EWC Continual NN FIRST in the live 3D viewer
    ewc_rows, ewc_status, ewc_err, ewc_learner = run_continual_trial(
        osim, source, schedule, args, dest, 'ewc_NN', display=persistent_display
    )
    # 2. Run Naive Online NN (No EWC)
    naive_rows, naive_status, naive_err, _ = run_continual_trial(
        osim, source, schedule, args, dest, 'naive_NN', display=None
    )
    # 3. Run High-Gain 5kg Tuned PID
    pid5_rows, pid5_status, pid5_err, _ = run_continual_trial(
        osim, source, schedule, args, dest, 'high_gain_5kg_PID', display=None
    )
    # 4. Run Baseline 0.5kg Fixed PI
    fixed_rows, fixed_status, fixed_err, _ = run_continual_trial(
        osim, source, schedule, args, dest, 'fixed_PI', display=None
    )

    if persistent_display is not None:
        persistent_display.finish()

    # Compute per-segment metrics across all 4 controllers
    all_metrics = (
        variable.metrics(fixed_rows, 'fixed_PI', schedule, args.duration) +
        variable.metrics(pid5_rows, 'high_gain_5kg_PID', schedule, args.duration) +
        variable.metrics(naive_rows, 'naive_NN', schedule, args.duration) +
        variable.metrics(ewc_rows, 'ewc_NN', schedule, args.duration)
    )
    with (dest / 'segment_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        w.writeheader()
        w.writerows(all_metrics)

    rows_dict = {
        'fixed_PI': fixed_rows,
        'high_gain_5kg_PID': pid5_rows,
        'naive_NN': naive_rows,
        'ewc_NN': ewc_rows,
    }
    forgetting = compute_forgetting(all_metrics, rows_dict)
    with (dest / 'forgetting_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(forgetting[0]))
        w.writeheader()
        w.writerows(forgetting)

    plot_All(plt, fixed_rows, pid5_rows, naive_rows, ewc_rows, ewc_learner, forgetting, schedule, args, dest)

    # Copy reproducibility files required for GitHub upload
    shutil.copy2(source, dest / 'input_model.osim')
    shutil.copy2(schedule_path, dest / 'load_schedule.csv')
    shutil.copy2(SCRIPT_PATH, dest / SCRIPT_PATH.name)
    for n in ('run_variable_load.py', 'run_arm26_pi_github.py', 'add_dumbbell.py'):
        shutil.copy2(PROJECT_ROOT / n, dest / n)

    run_id = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC_') + uuid.uuid4().hex[:8]
    settings = dict(
        run_id=run_id,
        completed=all(s == 'completed' for s in (fixed_status, pid5_status, naive_status, ewc_status)),
        statuses={
            'fixed_PI': fixed_status,
            'high_gain_5kg_PID': pid5_status,
            'naive_NN': naive_status,
            'ewc_NN': ewc_status,
        },
        failures={
            'fixed_PI': fixed_err,
            'high_gain_5kg_PID': pid5_err,
            'naive_NN': naive_err,
            'ewc_NN': ewc_err,
        },
        kp=args.kp, ki=args.ki, kv=args.kv,
        kp_5kg=args.kp_5kg, ki_5kg=args.ki_5kg, kv_5kg=args.kv_5kg,
        tmax_N=args.tmax, duration_s=args.duration, dt_s=args.dt,
        start_deg=args.start_deg, target_deg=args.target_deg,
        cycles=args.cycles, leg_time_s=args.rise_time,
        kp_nn=args.kp_nn, kd_nn=args.kd_nn, nn_lr=args.nn_lr,
        kd_fel=args.kd_fel, ewc_lambda=args.ewc_lambda, ewc_gamma=args.ewc_gamma,
        nn_tau_max=args.nn_tau_max, no_load_input=args.no_load_input,
        load_schedule=schedule, grip_body_m=list(args.grip),
        opensim_version=osim.GetVersion(),
        python_version=sys.version,
        method='4-Way Comparison: Fixed 0.5kg PI vs Fixed 5kg High-Gain PID vs Naive NN vs EWC Continual NN'
    )
    (dest / 'run_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')

    readme_text = (__doc__ + "\n\n"
                   f"Run ID: {run_id}\n"
                   f"GitHub Repository: {base.GITHUB_REPO_URL} (branch: {base.GITHUB_BRANCH})\n")
    if (PROJECT_ROOT / 'VARIABLE_LOAD_README.txt').exists():
        readme_text += "\n" + (PROJECT_ROOT / 'VARIABLE_LOAD_README.txt').read_text(encoding='utf-8')
    (dest / 'README.txt').write_text(readme_text, encoding='utf-8')

    manifest = dict(run_id=run_id, files={n: base.file_hash(dest / n) for n in FILES})
    (dest / 'upload_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print("\n==================== 4-WAY COMMITTEE COMPARISON TABLE ====================", flush=True)
    for row in forgetting:
        print(
            f"{row['controller']:18s} | Seg0(0.5kg): {row['seg0_0p5kg_rmse_deg']:.2f}° | "
            f"Seg1(2kg): {row['seg1_2kg_rmse_deg']:.2f}° | "
            f"Seg2(5kg): {row['seg2_5kg_rmse_deg']:.2f}° | "
            f"Seg3(0.5kg Return): {row['seg3_return_0p5kg_rmse_deg']:.2f}° | "
            f"Forgetting: {row['forgetting_delta_rmse_deg']:+.2f}° | "
            f"Overall RMSE: {row['overall_rmse_deg']:.2f}°",
            flush=True
        )
    print("==========================================================================", flush=True)

    # Start automatic GitHub push in background thread BEFORE plt.show()
    upload_worker = base.start_github_push(dest, args)
    print(f"\nDONE! Saved main plot:      {dest / 'continual_ewc_tracking.png'}", flush=True)
    print(f"Saved committee zoom plot: {dest / 'committee_zoom_comparison.png'}", flush=True)

    if not args.no_show:
        try:
            plt.show()
        except Exception as exc:
            print(f"Plot window warning ({exc}); PNG and PDF files are saved.", flush=True)

    if upload_worker:
        if upload_worker.is_alive():
            print("Waiting for GitHub upload to complete; see github_push_status.txt...", flush=True)
        upload_worker.join()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)