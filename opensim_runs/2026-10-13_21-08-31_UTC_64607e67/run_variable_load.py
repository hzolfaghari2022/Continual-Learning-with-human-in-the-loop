"""Time-varying handheld load: fixed PI versus manually scheduled PI.

Loads change at timed exchange events. Each event rebuilds physical mass/inertia
and transfers continuous joint/muscle state. This assumes load exchange at the
hand's current velocity without an impact impulse. It is not a variable-mass
flow model or an automatic/continually learning controller.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import uuid
import run_arm26_pi_github as base
from add_dumbbell import add_dumbbell, DEFAULT_GRIP

ROOT = Path(__file__).resolve().parent
FILES = ('fixed_PI_data.csv', 'scheduled_PI_data.csv', 'fixed_PI_motion.mot',
         'scheduled_PI_motion.mot', 'fixed_PI_states.sto', 'scheduled_PI_states.sto',
         'arm26_control_playback.osim', 'variable_load_plots.png', 'variable_load_plots.pdf',
         'segment_metrics.csv', 'README.txt', 'input_model.osim', 'run_settings.json',
         'load_schedule.csv', 'run_variable_load.py', 'run_arm26_pi_github.py', 'add_dumbbell.py')


def names(array):
    return [array.get(i) for i in range(array.getSize())]


def build(osim, source, load, args, previous=None, previous_model=None):
    model = osim.Model(str(source))
    if model.getControllerSet().getSize():
        raise ValueError('Use the original arm26_exosuit.osim without diagnostic controllers.')
    add_dumbbell(osim, model, load, args.grip)
    model.setName('arm26_variable_load_playback')
    model.setGravity(osim.Vec3(0, -9.80665, 0))
    shoulder = model.updCoordinateSet().get('r_shoulder_elev')
    shoulder.setDefaultValue(0); shoulder.setDefaultSpeedValue(0); shoulder.setDefaultLocked(True)
    elbow = model.updCoordinateSet().get('r_elbow_flex')
    elbow.setDefaultValue(math.radians(args.start_deg)); elbow.setDefaultSpeedValue(0); elbow.setDefaultLocked(False)
    muscles = model.updMuscles()
    if muscles.getSize() != 6 or model.getForceSet().getSize() != 7:
        raise ValueError('Expected six original muscles and the exo_flexor cable.')
    excitation = osim.PrescribedController()
    excitation.setName('fixed_muscle_excitation')
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        osim.Thelen2003Muscle.safeDownCast(muscle).setDefaultActivation(base.EXCITATION)
        excitation.addActuator(muscle)
        excitation.prescribeControlForActuator(muscle.getName(), osim.Constant(base.EXCITATION))
    model.addController(excitation)
    model.setUseVisualizer(False)
    state = model.initSystem()
    if previous is None:
        state.setTime(0); model.equilibrateMuscles(state)
    else:
        if names(model.getStateVariableNames()) != names(previous_model.getStateVariableNames()):
            raise RuntimeError('State layout changed during load exchange.')
        old = previous_model.getStateVariableValues(previous)
        model.setStateVariableValues(state, old)
        state.setTime(previous.getTime())
        new = model.getStateVariableValues(state)
        if any(abs(old.get(i)-new.get(i)) > 1e-12 for i in range(old.size())):
            raise RuntimeError('Load exchange did not preserve continuous state.')
    cable = osim.PathActuator.safeDownCast(model.updForceSet().get('exo_flexor'))
    if cable is None: raise ValueError('exo_flexor must be a PathActuator.')
    cable.overrideActuation(state, True)
    return model, state, elbow, cable


def run(osim, source, schedule, args, dest, scheduled):
    label = 'scheduled_PI' if scheduled else 'fixed_PI'
    current = 0; load = schedule[0]['load_kg']
    kp = schedule[0]['kp'] if scheduled else args.kp
    ki = schedule[0]['ki'] if scheduled else args.ki
    model, state, elbow, cable = build(osim, source, load, args)
    state_names = names(model.getStateVariableNames())
    # Store the integral contribution as torque, so changing Ki is explicit.
    integral_torque = 0.0
    display = None
    if args.live and (scheduled or getattr(args,"fixed_live",False)):
        try:
            display = base.LiveDisplay(osim, model, args, args.plt, state)
            args.display = display
        except Exception as exc:
            base.viewer_error(args, 'variable-load viewer', exc)
    rows, state_rows = [], []
    steps = round(args.duration / args.dt)
    for k in range(steps+1):
        t = state.getTime()
        if current+1 < len(schedule) and t >= schedule[current+1]['time_s'] - 1e-8:
            current += 1; task = schedule[current]
            model.realizeVelocity(state)
            error = base.reference(t, args.rise_time)-elbow.getValue(state)
            old_raw = kp*error + integral_torque
            if scheduled:
                kp, ki = task['kp'], task['ki']
                # Bumpless gain switching: raw torque stays continuous at the event.
                integral_torque = old_raw-kp*error
            old_model, old_state = model, state
            load = task['load_kg']
            model, state, elbow, cable = build(osim, source, load, args, old_state, old_model)
            if display:
                display.simulation_model = model
            print(f'{label}: t={t:.2f}s, load={load:g} kg, Kp={kp:g}, Ki={ki:g}', flush=True)
        model.realizeVelocity(state)
        q = elbow.getValue(state); speed = elbow.getSpeedValue(state)
        target = base.reference(t, args.rise_time); error = target-q
        arm = cable.computeMomentArm(state, elbow)
        if not 0 <= math.degrees(q) <= 90 or arm <= .005:
            raise RuntimeError(f'{label}: outside checked range at t={t:g}; angle={math.degrees(q):g}.')
        p = kp*error; raw = p+integral_torque
        tension = max(0.0,min(args.tmax,raw/arm)); applied=tension*arm
        can_integrate = not ((raw >= args.tmax*arm and error > 0) or (raw <= 0 and error < 0))
        cable.setOverrideActuation(state,tension); model.realizeDynamics(state)
        actual_tension = cable.getActuation(state)
        if abs(actual_tension-tension)>1e-6:raise RuntimeError('Cable force mismatch.')
        row=dict(time_s=t, load_kg=load, Kp=kp, Ki=ki, segment=current,
                 reference_deg=math.degrees(target), angle_deg=math.degrees(q),
                 error_deg=math.degrees(error), velocity_rad_s=speed,
                 tension_N=actual_tension, moment_arm_m=arm, P_Nm=p, I_Nm=integral_torque,
                 torque_raw_Nm=raw,torque_applied_Nm=applied,saturated=int(abs(raw-applied)>1e-9))
        muscles=model.getMuscles()
        for i in range(muscles.getSize()):
            muscle=muscles.get(i);n=muscle.getName()
            row[n+'_activation']=muscle.getActivation(state)
            row[n+'_fiber_length_m']=muscle.getFiberLength(state)
            row[n+'_tendon_force_N']=muscle.getTendonForce(state)
        if not all(math.isfinite(v) for v in row.values()):raise RuntimeError('Nonfinite result.')
        rows.append(row)
        vector=model.getStateVariableValues(state)
        state_rows.append([t]+[vector.get(i) for i in range(vector.size())])
        if display:
            try:
                display.update(state,row,final=k==steps)
                display.status.set_text(f'{label} | t={t:.2f}s | load={load:g} kg | Kp={kp:g}, Ki={ki:g} | error={row["error_deg"]:.2f} deg')
            except Exception as exc:
                base.viewer_error(args,'variable-load frame',exc);display.close();display=None
        if k==steps:break
        if can_integrate:integral_torque += ki*error*args.dt
        manager=osim.Manager(model);manager.setIntegratorAccuracy(1e-6)
        manager.setIntegratorMaximumStepSize(args.dt/5);manager.initialize(state)
        state=osim.State(manager.integrate((k+1)*args.dt))
    with (dest/(label+'_data.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    base.write_table(dest/(label+'_motion.mot'),['time','r_shoulder_elev','r_elbow_flex'],
                     [[r['time_s'],0,r['angle_deg']] for r in rows],True)
    base.write_table(dest/(label+'_states.sto'),['time']+state_names,state_rows)
    if scheduled or getattr(args,"fixed_live",False):
        playback=dest/'arm26_control_playback.osim'
        model.printToXML(str(playback))
        tree=ET.parse(playback)
        search=[ROOT/'Geometry', source.parent/'Geometry', source.parent]
        if args.geometry:search.insert(0,Path(args.geometry).expanduser().resolve())
        for node in tree.iter('mesh_file'):
            if node.text:
                ref=Path(node.text.strip())
                hit=next((d/ref for d in search if (d/ref).is_file()),None)
                if hit:node.text=str(hit.resolve())
        tree.write(playback,encoding='utf-8',xml_declaration=True)
    if display:display.finish()
    return rows


def metrics(rows,label,schedule,duration):
    result=[]
    for j, task in enumerate(schedule):
        end=schedule[j+1]['time_s'] if j+1<len(schedule) else duration+1e-8
        group=[r for r in rows if task['time_s']<=r['time_s']+1e-8 and r['time_s']<end-1e-8]
        if not group:continue
        result.append(dict(controller=label,segment=j,load_kg=task['load_kg'],
                           start_s=task['time_s'],end_s=min(end,duration),
                           rmse_deg=math.sqrt(sum(r['error_deg']**2 for r in group)/len(group)),
                           peak_abs_error_deg=max(abs(r['error_deg']) for r in group),
                           peak_tension_N=max(r['tension_N'] for r in group),
                           saturation_pct=100*sum(r['saturated'] for r in group)/len(group)))
    return result


def plot(plt,fixed,scheduled,schedule,args,dest):
    fig,axes=plt.subplots(5,1,figsize=(12,12),sharex=True,layout='constrained')
    t=[r['time_s'] for r in fixed]
    axes[0].step(t,[r['load_kg'] for r in fixed],where='post',color='#333333');axes[0].set_ylabel('Load (kg)')
    axes[1].plot(t,[r['reference_deg'] for r in fixed],'k--',label='Target')
    for rows,label,color in [(fixed,'Fixed PI','#d46b35'),(scheduled,'Scheduled PI','#087cab')]:
        axes[1].plot(t,[r['angle_deg'] for r in rows],label=label,color=color)
        axes[2].plot(t,[r['error_deg'] for r in rows],label=label,color=color)
        axes[3].plot(t,[r['tension_N'] for r in rows],label=label,color=color)
    axes[1].set_ylabel('Elbow (deg)');axes[2].set_ylabel('Error (deg)');axes[2].axhline(0,color='gray',lw=.7)
    axes[3].set_ylabel('Tension (N)');axes[3].axhline(args.tmax,color='black',ls=':',label='Tension limit')
    for key,style in [('Kp','-'),('Ki','--')]:
        axes[4].step(t,[r[key] for r in fixed],where='post',ls=style,color='#d46b35',label='Fixed '+key)
        axes[4].step(t,[r[key] for r in scheduled],where='post',ls=style,color='#087cab',label='Scheduled '+key)
    axes[4].set_ylabel('PI gains*');axes[4].set_xlabel('Simulation time (s)')
    for ax in axes:
        for task in schedule[1:]:ax.axvline(task['time_s'],color='gray',ls=':',alpha=.5)
        ax.grid(alpha=.2)
    for ax in axes[1:]:ax.legend(loc='upper right',ncol=3,fontsize=9)
    fig.suptitle('Changing handheld load: fixed PI vs manually scheduled PI\nSame trajectory, muscles, tension limit and load changes',fontsize=15)
    fig.supxlabel('*Kp: N m/rad; Ki: N m/(rad s). Scheduled gains are prescribed, not learned.',fontsize=9)
    fig.savefig(dest/'variable_load_plots.png',dpi=160);fig.savefig(dest/'variable_load_plots.pdf')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule',type=Path,default=ROOT/'variable_load_schedule.csv')
    parser.add_argument('--input',default=None);parser.add_argument('--geometry',default=None)
    parser.add_argument('--kp',type=float,default=30);parser.add_argument('--ki',type=float,default=15)
    parser.add_argument('--tmax',type=float,default=120,help='N; increased from original 80 N to study tuning separately from saturation.')
    parser.add_argument('--start-deg',type=float,default=20);parser.add_argument('--target-deg',type=float,default=80)
    parser.add_argument('--rise-time',type=float,default=4);parser.add_argument('--cycles',type=int,default=4)
    parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--grip',type=float,nargs=3,default=DEFAULT_GRIP)
    parser.add_argument('--no-show',action='store_true');parser.add_argument('--no-live',action='store_true')
    parser.add_argument('--no-push',action='store_true');parser.add_argument('--push-results',type=Path,default=None)
    args=parser.parse_args();base.GITHUB_FILES=FILES;base.GITHUB_RUNNER=__file__
    if args.push_results:
        if not base.push_completed_run(args.push_results.resolve()):raise SystemExit(2)
        return
    args.duration=.5+2*args.cycles*args.rise_time+1
    if not all(math.isfinite(x) for x in (args.dt,args.duration,args.kp,args.ki,args.tmax,args.start_deg,args.target_deg,*args.grip)):
        raise ValueError('All parameters must be finite.')
    if min(args.dt,args.rise_time,args.tmax,args.ki)<=0 or args.kp<0 or args.cycles<1 or not 0<=args.start_deg<args.target_deg<=90:
        raise ValueError('Invalid gains, times, capacity or angles.')
    with args.schedule.open(newline='',encoding='utf-8-sig') as f:
        schedule=[{k:float(r[k]) for k in ['time_s','load_kg','kp','ki']} for r in csv.DictReader(f)]
    if not schedule or schedule[0]['time_s']!=0:raise ValueError('Schedule must start at time 0.')
    for i,r in enumerate(schedule):
        if not all(math.isfinite(v) for v in r.values()) or min(r['load_kg'],r['kp'])<0 or r['ki']<=0:
            raise ValueError('Use finite nonnegative loads/Kp and positive Ki.')
        if r['time_s']>=args.duration or (i and r['time_s']<=schedule[i-1]['time_s']):raise ValueError('Times must increase within duration.')
        if abs(r['time_s']/args.dt-round(r['time_s']/args.dt))>1e-7:raise ValueError('Change times must be multiples of dt.')
    import opensim as osim
    osim.Logger.setLevelString('Warn')
    import matplotlib
    if args.no_show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.plt=plt;args.live=not(args.no_show or args.no_live)
    base.DEMO_START_DEG=args.start_deg;base.DEMO_TARGET_DEG=args.target_deg;base.DEMO_CYCLES=args.cycles
    dest=ROOT/('variable_load_results_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]);dest.mkdir()
    args.results_dir=dest
    source=base.find_input(ROOT,args.input)
    # Register geometry before constructing the physical models.
    base.prepare_display(osim,source,args.geometry,False)
    try:base.prepare_display(osim,source,args.geometry,args.live)
    except Exception as exc:base.viewer_error(args,'display setup',exc);args.live=False
    print('Results:',dest);print('Actuator input is tension in N, not current. Schedule:',schedule)
    fixed=run(osim,source,schedule,args,dest,False)
    scheduled=run(osim,source,schedule,args,dest,True)
    summaries=metrics(fixed,'fixed_PI',schedule,args.duration)+metrics(scheduled,'scheduled_PI',schedule,args.duration)
    with (dest/'segment_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    plot(plt,fixed,scheduled,schedule,args,dest)
    shutil.copy2(source,dest/'input_model.osim');shutil.copy2(args.schedule,dest/'load_schedule.csv')
    for name in ['run_variable_load.py','run_arm26_pi_github.py','add_dumbbell.py']:shutil.copy2(ROOT/name,dest/name)
    settings=dict(load_schedule=schedule,fixed_kp=args.kp,fixed_ki=args.ki,dt_s=args.dt,duration_s=args.duration,
                  reference_start_deg=args.start_deg,reference_target_deg=args.target_deg,cycles=args.cycles,
                  leg_time_s=args.rise_time,tmax_N=args.tmax,grip_body_m=args.grip,
                  completed=True,opensim_version=osim.GetVersion(),method='Manual gain scheduling, not continual learning')
    (dest/'run_settings.json').write_text(json.dumps(settings,indent=2))
    (dest/'README.txt').write_text(__doc__+'\n\n'+(ROOT/'VARIABLE_LOAD_README.txt').read_text())
    run_id=datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC_')+uuid.uuid4().hex[:8]
    (dest/'upload_manifest.json').write_text(json.dumps(dict(run_id=run_id,files={n:base.file_hash(dest/n) for n in FILES}),indent=2))
    for label,rows in [('Fixed PI',fixed),('Scheduled PI',scheduled)]:
        rmse=math.sqrt(sum(r['error_deg']**2 for r in rows)/len(rows))
        print(f'{label}: overall RMSE {rmse:.3f} deg; saturated {100*sum(r["saturated"] for r in rows)/len(rows):.1f}%')
    worker=base.start_github_push(dest,args)
    print('DONE:',dest/'variable_load_plots.png')
    if not args.no_show:plt.show()
    if hasattr(args,'display'):args.display.close()
    if worker:worker.join()


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
