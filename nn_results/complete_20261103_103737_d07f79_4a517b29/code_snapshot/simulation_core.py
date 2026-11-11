"""Physical OpenSim dynamics driven only by a trained neural force policy.
Every attempted episode and every sampled physical state are retained.
Launch via run_neural_controller.py; this module has no teacher-controller entry point.
"""
import runtime_setup
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import webbrowser
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parent
MUSCLES = ['TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA']
FEATURES = ['angle_rad', 'velocity_rad_s', 'reference_rad', 'reference_velocity_rad_s',
            'reference_acceleration_rad_s2', 'mass_kg', 'shoulder_angle_rad', 'previous_tension_N', 'error_integral_rad_s']
FEATURES += [m + '_activation' for m in MUSCLES]
FEATURES += [m + '_normalized_fiber_length' for m in MUSCLES]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from trajectories import reference, configure, FAMILIES


def build(o, cfg):
    from add_dumbbell import add_dumbbell, DEFAULT_GRIP
    o.ModelVisualizer.addDirToGeometrySearchPaths(str(ROOT/'Geometry'))
    model = o.Model(str(ROOT/'arm26_exosuit.osim'))
    if model.getControllerSet().getSize():
        raise ValueError('Expected original model without embedded controllers.')
    add_dumbbell(o, model, cfg['mass_kg'], DEFAULT_GRIP)
    model.setGravity(o.Vec3(0, -9.80665, 0))
    if [model.getCoordinateSet().get(i).getName() for i in range(model.getCoordinateSet().getSize())] != ['r_shoulder_elev', 'r_elbow_flex']:
        raise ValueError('Unexpected coordinate order; inverse-dynamics indexing must be reviewed.')
    el = model.updCoordinateSet().get('r_elbow_flex')
    sh = model.updCoordinateSet().get('r_shoulder_elev')
    sh.setDefaultValue(math.radians(cfg['shoulder_deg'])); sh.setDefaultSpeedValue(0); sh.setDefaultLocked(True)
    el.setDefaultValue(math.radians(cfg['start_deg'] + cfg['initial_offset_deg']))
    el.setDefaultSpeedValue(0); el.setDefaultLocked(False)
    ctrl = o.PrescribedController(); ctrl.setName('unchanged_fixed_muscle_excitation')
    mus = model.updMuscles()
    actual_names = [mus.get(i).getName() for i in range(mus.getSize())]
    if actual_names != MUSCLES:
        raise ValueError('Muscle order differs from the documented feature layout: ' + str(actual_names))
    for i in range(mus.getSize()):
        m = mus.get(i)
        m.setMaxIsometricForce(m.getMaxIsometricForce()*cfg['strength_scale'])
        o.Thelen2003Muscle.safeDownCast(m).setDefaultActivation(cfg['muscle_excitation'])
        ctrl.addActuator(m)
        ctrl.prescribeControlForActuator(m.getName(), o.Constant(cfg['muscle_excitation']))
    model.addController(ctrl); model.setUseVisualizer(False)
    state = model.initSystem(); state.setTime(0); model.equilibrateMuscles(state)
    cable = o.PathActuator.safeDownCast(model.updForceSet().get('exo_flexor'))
    if cable is None:
        raise ValueError('Missing exo_flexor PathActuator.')
    cable.overrideActuation(state, True); cable.setOverrideActuation(state, 0)
    return model, state, el, cable


def metrics(rows, cfg):
    def group(rs):
        if not rs:
            return None
        e = np.array([r['error_deg'] for r in rs]); f = np.array([r['tension_N'] for r in rs])
        return dict(rmse_deg=float(np.sqrt(np.mean(e*e))), mae_deg=float(np.mean(abs(e))),
                    max_abs_error_deg=float(max(abs(e))), peak_tension_N=float(max(f)),
                    mean_tension_N=float(np.mean(f)),
                    upper_saturation_percent=100*float(np.mean(f >= .999*cfg['force_cap_N'])),
                    clipped_percent=100*float(np.mean([r['saturated'] for r in rs])),
                    peak_abs_assistance_torque_Nm=float(max(abs(r['assistance_torque_Nm']) for r in rs)),
                    peak_actual_hand_speed_m_s=float(max(r['actual_hand_speed_m_s'] for r in rs)))
    return dict(all_samples=group(rows), motion_only=group([r for r in rows if r['moving']]))


def plot_episode(rows, dest, cfg, status):
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11})
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = [r['time_s'] for r in rows]
    ax[0].plot(t, [r['reference_deg'] for r in rows], 'k--', label='Desired')
    ax[0].plot(t, [r['angle_deg'] for r in rows], color='#0072b2', label='NN actual' if cfg.get('neural_checkpoint') else 'Compensated PID actual')
    ax[0].set_ylabel('Elbow angle (deg)'); ax[0].legend()
    ax[1].plot(t, [r['error_deg'] for r in rows], color='#d55e00'); ax[1].set_ylabel('Error (deg)')
    ax[2].plot(t, [r['tension_N'] for r in rows], color='#009e73')
    ax[2].axhline(cfg['force_cap_N'], color='black', ls=':', label='Cable limit')
    ax[2].set(ylabel='Cable tension (N)', xlabel='Time (s)'); ax[2].legend()
    for a in ax:
        a.grid(alpha=.2)
    fig.suptitle(f"{cfg['controller'] if cfg.get('neural_checkpoint') else 'Compensated PID'} | {cfg['episode_id']} | {cfg['mass_kg']:g} kg | peak hand speed {cfg['peak_hand_speed_m_s']:.3g} m/s\n" +
                 (f"NN-only direct cable force | {status}" if cfg.get('neural_checkpoint') else f"accel Kp={cfg['accel_kp']:g}, Ki={cfg['accel_ki']:g}, Kd={cfg['accel_kd']:g} | {status}"))
    fig.tight_layout(); fig.savefig(dest/'tracking.png', dpi=150); plt.close(fig)


def run_episode(dest):
    """One process per episode; preserve partial CSV even on interruption/failure."""
    import opensim as o
    from add_dumbbell import DEFAULT_GRIP
    o.Logger.setLevelString('Error')
    dest = Path(dest); cfg = read_json(dest/'settings.json')
    started = time.perf_counter(); rows = []; model = None; live_view = None; viewer_disabled = False
    result = dict(episode_id=cfg['episode_id'], task=cfg['task'], split=cfg['split'],
                  status='running', accepted_for_imitation=False, error='')
    write_json(dest/'status.json', result)
    try:
        model, state, el, cable = build(o, cfg)
        # A locked shoulder and one moving elbow give a constant station radius.
        probe = o.State(state); el.setSpeedValue(probe, 1.0); model.realizeVelocity(probe)
        v = model.getBodySet().get('r_ulna_radius_hand').findStationVelocityInGround(probe, o.Vec3(*DEFAULT_GRIP))
        radius = math.sqrt(sum(v.get(i)**2 for i in range(3)))
        configure(cfg,radius)
        cfg['hand_radius_m']=radius
        end=cfg['duration_s'];cfg['opensim_version']=o.GetVersion()
        model.realizeDynamics(state)
        from path_geometry import path_moment_arms
        from numpy_policy import NeuralCableController
        if not cfg.get('neural_checkpoint'):
            raise ValueError('NN-only simulation requires a trained checkpoint. No PID fallback exists.')
        controller=NeuralCableController(o,model,el,cable,cfg)
        cfg['controller']='neural_direct_force_no_teacher_fallback'
        write_json(dest/'settings.json', cfg)
        model.setName('arm26_nn_only_playback'); model.printToXML(str(dest/'playback_model.osim'))
        if cfg.get('live_viewer', True):
            try:
                from live_viewer import EpisodeViewer
                live_view=EpisodeViewer(o,model,state,ROOT)
                print('VIEWER ACTIVE: actual OpenSim states; one shared window across episodes.',flush=True)
            except Exception:
                (dest/'viewer_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                print('VIEWER FAILED: see viewer_error.txt; simulation continues.',flush=True)
        mus = model.getMuscles()
        names_arr = model.getStateVariableNames()
        state_names = [names_arr.get(i) for i in range(names_arr.getSize())]
        write_json(dest/'state_names.json', state_names)
        previous = 0.
        with (dest/'signals.csv').open('w', newline='', encoding='utf-8', buffering=1) as signal_file, \
             (dest/'states.csv').open('w', newline='', encoding='utf-8', buffering=1) as state_file:
            sw = None; stw = csv.writer(state_file); stw.writerow(['time_s']+state_names)
            steps = math.ceil(end/cfg['dt_s']); dt = cfg['dt_s']
            for k in range(steps+1):
                t = float(state.getTime()); h = min(dt, max(0., end-t))
                model.realizeVelocity(state)
                q, qd = el.getValue(state), el.getSpeedValue(state)
                qr, vr, ar, moving = reference(t, cfg)
                error = qr-q
                arms=path_moment_arms(o,model,state,el,cable,mus)
                arm=float(arms[0])
                if not 0 <= math.degrees(q) <= 90 or arm <= cfg['minimum_moment_arm_m']:
                    raise RuntimeError(f'Outside checked geometry: t={t:g}, q={math.degrees(q):g}, moment arm={arm:g}')
                command=controller.step(state,qr,vr,ar,h,arm)
                raw=command['torque_Nm'];F=command['force_N']
                cable.setOverrideActuation(state, F); model.realizeDynamics(state)
                actual = cable.getActuation(state)
                if abs(actual-F) > 1e-6:
                    raise RuntimeError('Applied cable force differs from requested force.')
                gripv = model.getBodySet().get('r_ulna_radius_hand').findStationVelocityInGround(state, o.Vec3(*DEFAULT_GRIP))
                r = dict(time_s=t, episode_id=cfg['episode_id'], task=cfg['task'], split=cfg['split'],
                         mass_kg=cfg['mass_kg'], shoulder_angle_rad=math.radians(cfg['shoulder_deg']), shoulder_deg=cfg['shoulder_deg'], trajectory=cfg['trajectory'], strength_scale=cfg['strength_scale'],
                         requested_peak_hand_speed_m_s=cfg['peak_hand_speed_m_s'],
                         reference_deg=math.degrees(qr), angle_deg=math.degrees(q), error_deg=math.degrees(error),
                         angle_rad=q, velocity_rad_s=qd, reference_rad=qr, reference_velocity_rad_s=vr,
                         reference_acceleration_rad_s2=ar, reference_hand_speed_m_s=radius*abs(vr),
                         actual_hand_speed_m_s=math.sqrt(sum(gripv.get(i)**2 for i in range(3))), moving=int(moving),
                         previous_tension_N=previous, tension_N=actual, force_cap_N=cfg['force_cap_N'],
                         moment_arm_m=arm, assistance_torque_Nm=actual*arm, raw_torque_Nm=raw,
                         error_integral_rad_s=command['error_integral_rad_s'], requested_force_N=command['requested_force_N'],
                         P_accel_rad_s2=command['P_accel'],I_accel_rad_s2=command['I_accel'],D_accel_rad_s2=command['D_accel'],
                         desired_acceleration_rad_s2=command['desired_acceleration_rad_s2'],
                         accel_Kp=cfg['accel_kp'],accel_Ki=cfg['accel_ki'],accel_Kd=cfg['accel_kd'],
                         saturated=int(abs(raw-actual*arm)>1e-8), muscle_excitation=cfg['muscle_excitation'])
                mtau = 0.
                for i, name in enumerate(MUSCLES):
                    m = mus.get(i); tendon = m.getTendonForce(state); moment = float(arms[i+1])
                    r[name+'_activation'] = m.getActivation(state)
                    r[name+'_fiber_length_m'] = m.getFiberLength(state)
                    r[name+'_normalized_fiber_length'] = m.getFiberLength(state)/m.getOptimalFiberLength()
                    r[name+'_fiber_velocity_m_s'] = m.getFiberVelocity(state)
                    r[name+'_path_length_m'] = m.getLength(state)
                    r[name+'_tendon_force_N'] = tendon
                    r[name+'_moment_arm_m'] = moment
                    r[name+'_torque_Nm'] = tendon*moment; mtau += tendon*moment
                r['muscle_torque_Nm'] = mtau
                if not all(math.isfinite(v) for v in r.values() if isinstance(v, (int,float))):
                    raise FloatingPointError('Nonfinite simulation record.')
                if sw is None:
                    sw = csv.DictWriter(signal_file, fieldnames=list(r)); sw.writeheader()
                sw.writerow(r); rows.append(r)
                if live_view is not None and not viewer_disabled:
                    try:live_view.update(state,force=(k==steps))
                    except Exception:
                        viewer_disabled=True
                        (dest/'viewer_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                        print('VIEWER CLOSED OR FAILED: simulation continues.',flush=True)
                sv = model.getStateVariableValues(state)
                stw.writerow([t]+[sv.get(i) for i in range(sv.size())])
                if k % max(1, round(.5/dt)) == 0:
                    signal_file.flush(); state_file.flush()
                if k == steps:
                    break
                manager = o.Manager(model); manager.setIntegratorAccuracy(cfg['integrator_accuracy'])
                manager.setIntegratorMaximumStepSize(h/5); manager.initialize(state)
                state = o.State(manager.integrate(t+h)); previous = actual
        result['status'] = 'completed'
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        (dest/'error.txt').write_text(result['error'], encoding='utf-8')
        if model is not None and 'state' in locals():
            try:
                sv=model.getStateVariableValues(state)
                values=[sv.get(i) for i in range(sv.size())]
                write_json(dest/'failure_terminal_state.json',dict(time_s=float(state.getTime()),
                           state_values=[v if math.isfinite(v) else str(v) for v in values]))
            except Exception:pass
    finally:
        if live_view is not None:live_view.close()
        result.update(metrics(rows, cfg)); m = result.get('motion_only')
        result['accepted_for_imitation'] = bool(result['status']=='completed' and m and
            m['rmse_deg']<=cfg['accept_rmse_deg'] and result['all_samples']['max_abs_error_deg']<=cfg['accept_max_error_deg'])
        result['controller_used']=cfg.get('controller','unknown')
        result['nn_outside_training_range_samples']=getattr(locals().get('controller'),'ood_samples',None)
        result['sample_count'] = len(rows); result['wall_seconds'] = time.perf_counter()-started
        result['last_recorded_time_s'] = rows[-1]['time_s'] if rows else 0
        try:
            if rows:
                X = np.array([[r[n] for n in FEATURES] for r in rows], dtype=np.float64)
                Y = np.array([[r['tension_N']] for r in rows], dtype=np.float64)
                np.savez_compressed(dest/'training_arrays.npz', X_raw=X, y_force_N=Y,
                                    feature_names=np.array(FEATURES), time_s=np.array([r['time_s'] for r in rows]))
                with (dest/'states.csv').open(encoding='utf-8') as f:
                    sr=list(csv.reader(f))
                with (dest/'states.sto').open('w',encoding='utf-8') as f:
                    f.write(f"Arm26 PID states\nversion=1\nnRows={len(sr)-1}\nnColumns={len(sr[0])}\ninDegrees=no\nendheader\n")
                    f.write('\t'.join(['time']+sr[0][1:])+'\n')
                    for rr in sr[1:]:f.write('\t'.join(rr)+'\n')
                if cfg.get('save_episode_plots', True):
                    plot_episode(rows, dest, cfg, 'ACCEPTED' if result['accepted_for_imitation'] else 'NOT ACCEPTED')
        except Exception:
            result['export_error'] = traceback.format_exc()
            result['accepted_for_imitation'] = False
        write_json(dest/'status.json', result)
    return 0 if result['status']=='completed' and 'export_error' not in result else 1


if __name__=="__main__":
    raise SystemExit("Run run_neural_controller.py to start the neural OpenSim simulation.")
