"""Continual Learning (Online Neural Correction + EWC) for PI-controlled Arm26.

Runs and compares three controllers across the changing dumbbell schedule
(default: heavy_load_schedule.csv: 0.5 -> 2 -> 5 -> 0.5 kg over 65.5 s):
  1. Fixed PI (baseline, no adaptation)
  2. Naive Online NN + PI (learns online, no EWC protection -> forgets 0.5 kg)
  3. EWC Continual NN + PI (consolidates Fisher information at load switches)

Automatically saves all CSV/MOT/STO/PNG/PDF artifacts and pushes the completed
run snapshot to:
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
import traceback
import uuid
import xml.etree.ElementTree as ET
import numpy as np

import run_variable_load as variable
import run_arm26_pi_github as base
from add_dumbbell import DEFAULT_GRIP

ROOT = Path(__file__).resolve().parent

# Complete file list verified by base.push_completed_run() before GitHub upload
FILES = (
    'fixed_PI_data.csv',
    'naive_NN_data.csv',
    'ewc_NN_data.csv',
    'fixed_PI_motion.mot',
    'naive_NN_motion.mot',
    'ewc_NN_motion.mot',
    'fixed_PI_states.sto',
    'naive_NN_states.sto',
    'ewc_NN_states.sto',
    'arm26_control_playback.osim',
    'continual_ewc_tracking.png',
    'continual_ewc_tracking.pdf',
    'fisher_importance.png',
    'dumbbell_weight.png',
    'segment_metrics.csv',
    'forgetting_summary.csv',
    'input_model.osim',
    'load_schedule.csv',
    'run_settings.json',
    'README.txt',
    'run_continual_ewc.py',
    'run_variable_load.py',
    'run_arm26_pi_github.py',
    'add_dumbbell.py',
)


class ResidualMLPWithEWC:
    """Two-layer Tanh MLP outputting residual elbow torque tau_NN (N m) with Online EWC."""

    def __init__(self, in_dim=8, hidden_dim=24, lr=0.010, ewc_lambda=250.0,
                 gamma=0.95, tau_limit=30.0, seed=42):
        rng = np.random.default_rng(seed)
        # Hidden layer: Xavier init. Output layer: ZERO init so tau_NN = 0 at t = 0.
        self.W1 = rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(hidden_dim, in_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.W2 = np.zeros((1, hidden_dim), dtype=float)
        self.b2 = np.zeros(1, dtype=float)

        self.lr = lr
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.tau_limit = tau_limit

        # Online EWC memory: anchor weights (theta*) and running diagonal Fisher (F)
        self.consolidated = False
        self.anchor = {}
        self.fisher = {
            'W1': np.zeros_like(self.W1),
            'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2),
            'b2': np.zeros_like(self.b2),
        }
        self.segment_features = []
        self.fisher_history = []

    def _forward_with_cache(self, x):
        z1 = self.W1 @ x + self.b1
        h1 = np.tanh(z1)
        out = float((self.W2 @ h1 + self.b2)[0])
        # Smoothly bound residual torque to [-tau_limit, +tau_limit]
        tau_nn = self.tau_limit * math.tanh(out / self.tau_limit)
        scale = 1.0 - (tau_nn / self.tau_limit) ** 2  # d(tau_nn)/d(out)
        return tau_nn, (x, z1, h1, scale)

    def predict(self, x):
        tau_nn, _ = self._forward_with_cache(x)
        return tau_nn

    def _output_gradients(self, cache):
        """Exact analytical gradient d(tau_NN)/d(theta) for every network weight."""
        x, z1, h1, scale = cache
        d_out = scale
        dW2 = d_out * h1.reshape(1, -1)
        db2 = np.array([d_out], dtype=float)
        dh1 = (self.W2[0] * d_out) * (1.0 - h1 ** 2)
        dW1 = np.outer(dh1, x)
        db1 = dh1
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def step_update(self, x, teaching_signal, can_learn=True):
        """
        Online Feedback Error Learning (FEL) + EWC update step.
        teaching_signal: error-driven torque adjustment (N m).
        """
        tau_nn, cache = self._forward_with_cache(x)
        self.segment_features.append(x.copy())

        if not can_learn or self.lr <= 0.0:
            return tau_nn, 0.0

        grads = self._output_gradients(cache)
        ewc_penalty = 0.0

        for name, param in [('W1', self.W1), ('b1', self.b1),
                            ('W2', self.W2), ('b2', self.b2)]:
            # 1. Task gradient from Feedback Error Learning
            g_task = -teaching_signal * grads[name]

            # 2. EWC quadratic spring gradient: lambda * F_i * (theta_i - theta*_i)
            g_ewc = 0.0
            if self.consolidated and self.ewc_lambda > 0.0:
                diff = param - self.anchor[name]
                g_ewc = self.ewc_lambda * self.fisher[name] * diff
                ewc_penalty += 0.5 * self.ewc_lambda * float(np.sum(self.fisher[name] * (diff ** 2)))

            g_total = g_task + g_ewc
            np.clip(g_total, -5.0, 5.0, out=g_total)
            param -= self.lr * g_total

        return tau_nn, ewc_penalty

    def consolidate_segment(self, seg_idx, load_kg):
        """
        Compute Fisher Information Matrix diagonal F_i = E[(d(tau_NN)/d(theta_i))^2]
        at the end of a load segment and update the EWC anchor weights.
        """
        if not self.segment_features:
            return
        # Use second half of segment where trajectory tracking has settled
        half = max(1, len(self.segment_features) // 2)
        samples = self.segment_features[-half:]
        N = len(samples)

        seg_fisher = {k: np.zeros_like(v) for k, v in self.fisher.items()}
        for x in samples:
            _, cache = self._forward_with_cache(x)
            grads = self._output_gradients(cache)
            for k in seg_fisher:
                seg_fisher[k] += (grads[k] ** 2) / N

        # Normalize Fisher so ewc_lambda is well-conditioned across segments
        max_f = max(float(np.max(v)) for v in seg_fisher.values())
        if max_f > 1e-12:
            for k in seg_fisher:
                seg_fisher[k] /= max_f

        for name, param in [('W1', self.W1), ('b1', self.b1),
                            ('W2', self.W2), ('b2', self.b2)]:
            if self.consolidated:
                self.fisher[name] = self.gamma * self.fisher[name] + seg_fisher[name]
            else:
                self.fisher[name] = seg_fisher[name]
            self.anchor[name] = param.copy()

        self.fisher_history.append({
            'segment': seg_idx,
            'load_kg': load_kg,
            'W1_mean_fisher': float(np.mean(self.fisher['W1'])),
            'W2_mean_fisher': float(np.mean(self.fisher['W2'])),
            'W1_max_fisher': float(np.max(self.fisher['W1'])),
            'W2_max_fisher': float(np.max(self.fisher['W2'])),
            'W1_fisher_ snapshot': self.fisher['W1'].copy(),
        })
        self.consolidated = True
        self.segment_features.clear()


def make_features(q, speed, target, error, load_kg, use_load_input=True):
    """Construct 8D normalized state feature vector for the neural network."""
    load_feat = (load_kg / 5.0) if use_load_input else 0.0
    return np.array([
        1.0,
        math.sin(q),
        math.cos(q),
        math.sin(target),
        math.cos(target),
        speed / 2.0,
        error,
        load_feat
    ], dtype=float)


def run_continual_trial(osim, source, schedule, args, dest, mode):
    """
    Simulate one controller across the entire variable-load schedule:
      - 'fixed_PI' : Baseline fixed PI (tau_NN = 0)
      - 'naive_NN' : Online NN + PI without EWC (ewc_lambda = 0)
      - 'ewc_NN'   : Online NN + PI with EWC consolidation at load transitions
    """
    current = 0
    load = schedule[0]['load_kg']
    kp, ki = args.kp, args.ki
    model, state, elbow, cable = variable.build(osim, source, load, args)
    state_names = variable.names(model.getStateVariableNames())
    integral_torque = 0.0
    prev_error = 0.0

    nn_learner = None
    if mode in ('naive_NN', 'ewc_NN'):
        lam = args.ewc_lambda if mode == 'ewc_NN' else 0.0
        nn_learner = ResidualMLPWithEWC(
            in_dim=8,
            hidden_dim=args.hidden_dim,
            lr=args.nn_lr,
            ewc_lambda=lam,
            gamma=args.ewc_gamma,
            tau_limit=args.nn_tau_max,
            seed=args.seed
        )

    display = None
    if args.live and mode == 'ewc_NN':
        try:
            display = base.LiveDisplay(osim, model, args, args.plt, state)
            args.display = display
        except Exception as exc:
            base.viewer_error(args, 'continual-ewc viewer', exc)

    rows, state_rows = [], []
    steps = round(args.duration / args.dt)
    status = 'completed'
    failure_reason = ''
    print(f"\n--- Running [{mode}] | Kp={kp:g}, Ki={ki:g}, Kv={args.kv:g}, Tmax={args.tmax:g} N ---", flush=True)

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
                nn_learner.segment_features.clear()

            current += 1
            task = schedule[current]
            old_model, old_state = model, state
            load = task['load_kg']
            model, state, elbow, cable = variable.build(osim, source, load, args, old_state, old_model)
            if display:
                display.simulation_model = model
            print(f"  {mode}: t={t:.2f}s -> switched dumbbell to {load:g} kg (Segment {current})", flush=True)

        model.realizeVelocity(state)
        q = elbow.getValue(state)
        speed = elbow.getSpeedValue(state)
        target = base.reference(t, args.rise_time)
        error = target - q
        edot = 0.0 if k == 0 else (error - prev_error) / args.dt
        prev_error = error
        arm = cable.computeMomentArm(state, elbow)

        if not 0 <= math.degrees(q) <= 90 or arm <= 0.005:
            status = 'FAILED_OUT_OF_RANGE'
            failure_reason = (f"{mode} left verified [0, 90] deg range at t={t:.3f}s "
                              f"(angle={math.degrees(q):.2f} deg, load={load:g} kg).")
            print(f"  WARNING: {failure_reason} Saving recorded states and continuing.", flush=True)
            break

        # 1. Baseline Fixed PI torque (plus optional Kv velocity damping if > 0)
        p = kp * error
        damping = -args.kv * speed
        tau_pi = p + integral_torque + damping

        # 2. Continual Neural Correction torque tau_NN
        tau_nn = 0.0
        ewc_pen = 0.0
        if nn_learner is not None:
            x_feat = make_features(q, speed, target, error, load, use_load_input=not args.no_load_input)
            tau_nn = nn_learner.predict(x_feat)

        tau_raw = tau_pi + tau_nn
        tension = max(0.0, min(args.tmax, tau_raw / arm))
        applied = tension * arm

        # Conditional integration & learning freeze under saturation
        can_integrate = not ((tau_raw >= args.tmax * arm and error > 0) or (tau_raw <= 0 and error < 0))

        # 3. Online weight update via Feedback Error Learning + EWC
        if nn_learner is not None and k < steps:
            # PD error signal teaches the feedforward/residual network to cancel lag & sag
            pd_error_signal = kp * error + args.kd_fel * edot
            teaching_signal = args.fel_gain * pd_error_signal * args.dt
            _, ewc_pen = nn_learner.step_update(x_feat, teaching_signal, can_learn=can_integrate)

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

        if display:
            try:
                display.update(state, row, final=k == steps)
                display.status.set_text(
                    f"{mode} | t={t:.2f}s | load={load:g}kg | err={row['error_deg']:.2f}° | "
                    f"tau_NN={tau_nn:.2f}Nm"
                )
            except Exception as exc:
                base.viewer_error(args, 'continual-ewc frame', exc)
                display.close()
                display = None

        if k == steps:
            break
        if can_integrate:
            integral_torque += ki * error * args.dt

        manager = osim.Manager(model)
        manager.setIntegratorAccuracy(1e-6)
        manager.setIntegratorMaximumStepSize(args.dt / 5)
        manager.initialize(state)
        state = osim.State(manager.integrate((k + 1) * args.dt))

    # Consolidate final segment for Fisher importance visualization
    if nn_learner is not None and mode == 'ewc_NN' and nn_learner.segment_features:
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
        search = [ROOT / 'Geometry', source.parent / 'Geometry', source.parent]
        if args.geometry:
            search.insert(0, Path(args.geometry).expanduser().resolve())
        for node in tree.iter('mesh_file'):
            if node.text:
                ref = Path(node.text.strip())
                hit = next((d / ref for d in search if (d / ref).is_file()), None)
                if hit:
                    node.text = str(hit.resolve())
        tree.write(playback, encoding='utf-8', xml_declaration=True)

    if display:
        display.finish()
    return rows, status, failure_reason, nn_learner


def compute_forgetting(all_metrics):
    """Compare Segment 0 (initial 0.5 kg) vs final return to 0.5 kg after heavy loads."""
    summary = []
    for ctrl in ('fixed_PI', 'naive_NN', 'ewc_NN'):
        segs = [m for m in all_metrics if m['controller'] == ctrl]
        if not segs:
            continue
        first_rmse = segs[0]['rmse_deg']
        return_rmse = segs[-1]['rmse_deg'] if len(segs) >= 4 else float('nan')
        return_peak = segs[-1]['peak_abs_error_deg'] if len(segs) >= 4 else float('nan')
        delta_rmse = (return_rmse - first_rmse) if len(segs) >= 4 else float('nan')
        summary.append(dict(
            controller=ctrl,
            segments_completed=len(segs),
            initial_load_kg=segs[0]['load_kg'],
            seg0_rmse_deg=first_rmse,
            seg1_2kg_rmse_deg=segs[1]['rmse_deg'] if len(segs) > 1 else float('nan'),
            seg2_5kg_rmse_deg=segs[2]['rmse_deg'] if len(segs) > 2 else float('nan'),
            return_seg_rmse_deg=return_rmse,
            forgetting_delta_rmse_deg=delta_rmse,
            return_seg_peak_error_deg=return_peak,
            overall_rmse_deg=math.sqrt(sum(m['rmse_deg'] ** 2 for m in segs) / len(segs))
        ))
    return summary


def plot_All(plt, fixed, naive, ewc, ewc_learner, schedule, args, dest):
    # 1. Dumbbell weight profile plot
    t_fixed = [r['time_s'] for r in fixed]
    fig_w, ax_w = plt.subplots(figsize=(11, 3), layout='constrained')
    ax_w.step(t_fixed, [r['load_kg'] for r in fixed], where='post', color='#333333', lw=2)
    ax_w.set(xlabel='Simulation time (s)', ylabel='Dumbbell mass (kg)',
             title='Physical dumbbell mass during the Continual Learning experiment')
    ax_w.grid(alpha=0.25)
    fig_w.savefig(dest / 'dumbbell_weight.png', dpi=160)
    plt.close(fig_w)

    # 2. Main 5-panel Continual Learning comparison figure
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True, layout='constrained')
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
        (fixed, 'Fixed PI (no learning)', '#b5651d', 1.3, 0.85),
        (naive, 'Naive Online NN + PI (no EWC)', '#d62728', 1.4, 0.85),
        (ewc, 'EWC Continual NN + PI (proposed)', '#0072b2', 2.0, 1.0),
    ]
    for rows, label, color, lw, alpha in traces:
        if not rows:
            continue
        tr = [r['time_s'] for r in rows]
        axes[1].plot(tr, [r['angle_deg'] for r in rows], color=color, lw=lw, alpha=alpha, label=label)
        axes[2].plot(tr, [r['error_deg'] for r in rows], color=color, lw=lw, alpha=alpha, label=label)
        axes[3].plot(tr, [r['NN_Nm'] for r in rows], color=color, lw=lw, alpha=alpha, label=label)
        axes[4].plot(tr, [r['tension_N'] for r in rows], color=color, lw=lw, alpha=alpha, label=label)

    axes[1].set_ylabel('Elbow (deg)')
    axes[1].legend(loc='upper right', ncol=4, fontsize=8.5)
    axes[1].set_title('2. Joint Angle Tracking', loc='left', weight='bold')

    axes[2].axhline(0, color='gray', lw=0.8)
    axes[2].set_ylabel('Error (deg)')
    axes[2].legend(loc='upper right', ncol=3, fontsize=8.5)
    axes[2].set_title('3. Tracking Error (Target - Actual)', loc='left', weight='bold')

    axes[3].axhline(0, color='gray', lw=0.7, ls='--')
    axes[3].set_ylabel('NN Torque (N·m)')
    axes[3].legend(loc='upper right', ncol=3, fontsize=8.5)
    axes[3].set_title('4. Learned Neural Correction Torque (tau_NN)', loc='left', weight='bold')

    axes[4].axhline(args.tmax, color='black', ls=':', lw=1.5, label=f'Limit ({args.tmax:g} N)')
    axes[4].set_ylabel('Tension (N)')
    axes[4].set_xlabel('Simulation time (s)')
    axes[4].legend(loc='upper right', ncol=4, fontsize=8.5)
    axes[4].set_title('5. Applied Flexor Cable Tension', loc='left', weight='bold')

    for ax in axes:
        ax.grid(alpha=0.22)
        ax.set_xlim(0, args.duration)
        for task in schedule[1:]:
            ax.axvline(task['time_s'], color='#444444', ls=':', lw=1.0)

    fig.suptitle(
        f"Arm26 Continual Learning: Fixed PI vs. Naive Online NN vs. EWC-Protected NN\n"
        f"Kp={args.kp:g}, Ki={args.ki:g}, Kv={args.kv:g} | EWC lambda={args.ewc_lambda:g} | Cable Limit={args.tmax:g} N",
        fontsize=14, weight='bold'
    )
    fig.savefig(dest / 'continual_ewc_tracking.png', dpi=170)
    fig.savefig(dest / 'continual_ewc_tracking.pdf')

    # 3. Fisher Information Matrix Importance Heatmap
    fig_f, ax_f = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    if ewc_learner is not None and ewc_learner.consolidated:
        im = ax_f[0].imshow(ewc_learner.fisher['W1'], aspect='auto', cmap='viridis')
        fig_f.colorbar(im, ax=ax_f[0], label='Normalized Fisher Importance $F_i$')
        feat_names = ['Bias', 'sin(q)', 'cos(q)', 'sin(qref)', 'cos(qref)', 'speed', 'error', 'load']
        ax_f[0].set_xticks(range(len(feat_names)), feat_names, rotation=30, ha='right')
        ax_f[0].set_ylabel('Hidden Neuron Index')
        ax_f[0].set_title('EWC Consolidated Fisher Matrix ($W_1$)')

        w2_f = ewc_learner.fisher['W2'][0]
        ax_f[1].bar(range(len(w2_f)), w2_f, color='#0072b2')
        ax_f[1].set_xlabel('Hidden Neuron Index')
        ax_f[1].set_ylabel('Normalized Fisher Importance $F_i$')
        ax_f[1].set_title('Output Layer ($W_2$) Parameter Importance')
        ax_f[1].grid(alpha=0.25)
    else:
        ax_f[0].text(0.5, 0.5, 'No Fisher consolidation recorded', ha='center')
    fig_f.suptitle('Elastic Weight Consolidation (EWC) Parameter Importance Across Dumbbell Tasks', weight='bold')
    fig_f.savefig(dest / 'fisher_importance.png', dpi=160)
    plt.close(fig_f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule', type=Path, default=ROOT / 'heavy_load_schedule.csv')
    p.add_argument('--input', default=None)
    p.add_argument('--geometry', default=None)
    p.add_argument('--kp', type=float, default=30.0)
    p.add_argument('--ki', type=float, default=15.0)
    p.add_argument('--kv', type=float, default=0.0, help='Default 0.0 (pure PI). Set e.g. 2.0 for optional velocity damping.')
    p.add_argument('--tmax', type=float, default=240.0, help='Cable tension cap (N). 240 N prevents 5 kg saturation.')
    p.add_argument('--start-deg', type=float, default=20.0)
    p.add_argument('--target-deg', type=float, default=80.0)
    p.add_argument('--rise-time', type=float, default=4.0)
    p.add_argument('--cycles', type=int, default=8)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--grip', type=float, nargs=3, default=DEFAULT_GRIP)
    # Continual Learning (FEL + EWC) parameters
    p.add_argument('--hidden-dim', type=int, default=24)
    p.add_argument('--nn-lr', type=float, default=0.010, help='Online learning rate for neural correction.')
    p.add_argument('--fel-gain', type=float, default=1.6, help='Feedback Error Learning rate.')
    p.add_argument('--kd-fel', type=float, default=12.0, help='Error-derivative damping inside FEL teaching signal.')
    p.add_argument('--ewc-lambda', type=float, default=250.0, help='EWC regularization strength.')
    p.add_argument('--ewc-gamma', type=float, default=0.95, help='Online EWC Fisher decay across segments.')
    p.add_argument('--nn-tau-max', type=float, default=30.0, help='Max magnitude of neural torque correction (N m).')
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
    base.GITHUB_RUNNER = __file__

    if args.push_results:
        folder = args.push_results.expanduser()
        if not folder.is_absolute():
            folder = ROOT / folder
        if not base.push_completed_run(folder.resolve()):
            raise SystemExit(2)
        return

    args.duration = 0.5 + 2 * args.cycles * args.rise_time + 1.0
    with args.schedule.open(newline='', encoding='utf-8-sig') as f:
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

    dest = ROOT / ('continual_ewc_results_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:4])
    dest.mkdir()
    args.results_dir = dest
    source = base.find_input(ROOT, args.input)
    base.prepare_display(osim, source, args.geometry, False)
    try:
        base.prepare_display(osim, source, args.geometry, args.live)
    except Exception as exc:
        base.viewer_error(args, 'display setup', exc)
        args.live = False

    print(f"ARM26 CONTINUAL LEARNING (RESIDUAL NN + EWC) RUNNER\nInput: {source}\nResults: {dest}", flush=True)

    # Run all 3 controllers sequentially
    fixed_rows, fixed_status, fixed_err, _ = run_continual_trial(osim, source, schedule, args, dest, 'fixed_PI')
    naive_rows, naive_status, naive_err, _ = run_continual_trial(osim, source, schedule, args, dest, 'naive_NN')
    ewc_rows, ewc_status, ewc_err, ewc_learner = run_continual_trial(osim, source, schedule, args, dest, 'ewc_NN')

    # Compute per-segment metrics
    all_metrics = (
        variable.metrics(fixed_rows, 'fixed_PI', schedule, args.duration) +
        variable.metrics(naive_rows, 'naive_NN', schedule, args.duration) +
        variable.metrics(ewc_rows, 'ewc_NN', schedule, args.duration)
    )
    with (dest / 'segment_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        w.writeheader()
        w.writerows(all_metrics)

    forgetting = compute_forgetting(all_metrics)
    with (dest / 'forgetting_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(forgetting[0]))
        w.writeheader()
        w.writerows(forgetting)

    plot_All(plt, fixed_rows, naive_rows, ewc_rows, ewc_learner, schedule, args, dest)

    # Copy reproducibility files required for GitHub upload
    shutil.copy2(source, dest / 'input_model.osim')
    shutil.copy2(args.schedule, dest / 'load_schedule.csv')
    shutil.copy2(Path(__file__).resolve(), dest / 'run_continual_ewc.py')
    for n in ('run_variable_load.py', 'run_arm26_pi_github.py', 'add_dumbbell.py'):
        shutil.copy2(ROOT / n, dest / n)

    run_id = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC_') + uuid.uuid4().hex[:8]
    settings = dict(
        run_id=run_id,
        completed=(fixed_status == 'completed' and naive_status == 'completed' and ewc_status == 'completed'),
        statuses={'fixed_PI': fixed_status, 'naive_NN': naive_status, 'ewc_NN': ewc_status},
        failures={'fixed_PI': fixed_err, 'naive_NN': naive_err, 'ewc_NN': ewc_err},
        kp=args.kp, ki=args.ki, kv=args.kv, tmax_N=args.tmax,
        duration_s=args.duration, dt_s=args.dt,
        start_deg=args.start_deg, target_deg=args.target_deg,
        cycles=args.cycles, leg_time_s=args.rise_time,
        hidden_dim=args.hidden_dim, nn_lr=args.nn_lr,
        fel_gain=args.fel_gain, kd_fel=args.kd_fel,
        ewc_lambda=args.ewc_lambda, ewc_gamma=args.ewc_gamma,
        nn_tau_max=args.nn_tau_max, no_load_input=args.no_load_input,
        load_schedule=schedule, grip_body_m=list(args.grip),
        opensim_version=osim.GetVersion(),
        python_version=sys.version,
        method='Residual MLP + Feedback Error Learning (FEL) + Online Elastic Weight Consolidation (EWC)'
    )
    (dest / 'run_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')

    readme_text = (__doc__ + "\n\n"
                   f"Run ID: {run_id}\n"
                   f"GitHub Repository: {base.GITHUB_REPO_URL} (branch: {base.GITHUB_BRANCH})\n"
                   f"Statuses: fixed_PI={fixed_status}, naive_NN={naive_status}, ewc_NN={ewc_status}\n")
    if (ROOT / 'VARIABLE_LOAD_README.txt').exists():
        readme_text += "\n" + (ROOT / 'VARIABLE_LOAD_README.txt').read_text(encoding='utf-8')
    (dest / 'README.txt').write_text(readme_text, encoding='utf-8')

    manifest = dict(run_id=run_id, files={n: base.file_hash(dest / n) for n in FILES})
    (dest / 'upload_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print("\n================ CONTINUAL LEARNING SUMMARY ================", flush=True)
    for row in forgetting:
        print(
            f"{row['controller']:10s} | Seg0 (0.5kg): {row['seg0_rmse_deg']:.3f}° | "
            f"Seg1 (2kg): {row['seg1_2kg_rmse_deg']:.3f}° | "
            f"Seg2 (5kg): {row['seg2_5kg_rmse_deg']:.3f}° | "
            f"Seg3 Return (0.5kg): {row['return_seg_rmse_deg']:.3f}° | "
            f"Forgetting Delta: {row['forgetting_delta_rmse_deg']:+.3f}° | "
            f"Overall RMSE: {row['overall_rmse_deg']:.3f}°",
            flush=True
        )
    print("============================================================", flush=True)

    # Start automatic GitHub push in background thread before plt.show()
    upload_worker = base.start_github_push(dest, args)
    print(f"\nDONE! Plots saved to: {dest / 'continual_ewc_tracking.png'}", flush=True)

    if not args.no_show:
        try:
            plt.show()
        except Exception as exc:
            print(f"Plot window unavailable ({exc}); PNG and PDF files are saved.", flush=True)

    if hasattr(args, 'display'):
        args.display.close()
    plt.close('all')

    # Wait for GitHub push to finish so VS Code / terminal doesn't exit early
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