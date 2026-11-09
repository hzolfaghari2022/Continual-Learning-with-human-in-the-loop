"""Longer heavy-load experiment with FIXED PI gains throughout.

0.5 -> 2 -> 5 -> 0.5 kg over 65.5 s. Default Kp=30, Ki=15, tension cap=120 N.
Load exchanges update physical mass and inertia, with no idealized impact.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
from datetime import datetime, timezone
import uuid
import sys
import run_variable_load as variable
import run_arm26_pi_github as base
from add_dumbbell import DEFAULT_GRIP

ROOT=Path(__file__).resolve().parent
FILES=('fixed_PI_data.csv','fixed_PI_motion.mot','fixed_PI_states.sto',
       'arm26_control_playback.osim','fixed_PI_tracking.png','fixed_PI_tracking.pdf',
       'dumbbell_weight.png','segment_metrics.csv','input_model.osim','run_settings.json',
       'load_schedule.csv','README.txt','run_heavy_load_fixed_PI.py','run_variable_load.py',
       'run_arm26_pi_github.py','add_dumbbell.py')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schedule',type=Path,default=ROOT/'heavy_load_schedule.csv')
    p.add_argument('--input',default=None);p.add_argument('--geometry',default=None)
    p.add_argument('--kp',type=float,default=30);p.add_argument('--ki',type=float,default=15)
    p.add_argument('--tmax',type=float,default=120)
    p.add_argument('--start-deg',type=float,default=20);p.add_argument('--target-deg',type=float,default=80)
    p.add_argument('--rise-time',type=float,default=4);p.add_argument('--cycles',type=int,default=8)
    p.add_argument('--dt',type=float,default=.01);p.add_argument('--grip',type=float,nargs=3,default=DEFAULT_GRIP)
    p.add_argument('--no-show',action='store_true');p.add_argument('--no-live',action='store_true')
    p.add_argument('--no-push',action='store_true');p.add_argument('--push-results',type=Path,default=None)
    args=p.parse_args();base.GITHUB_FILES=FILES;base.GITHUB_RUNNER=__file__
    if args.push_results:
        if not base.push_completed_run(args.push_results.resolve()):raise SystemExit(2)
        return
    args.duration=.5+2*args.cycles*args.rise_time+1
    if not all(math.isfinite(v) for v in (args.kp,args.ki,args.dt,args.duration,args.tmax,args.start_deg,args.target_deg,*args.grip)):
        raise ValueError('All parameters must be finite.')
    if min(args.dt,args.rise_time,args.tmax,args.ki)<=0 or args.kp<0 or args.cycles<1 or not 0<=args.start_deg<args.target_deg<=90:
        raise ValueError('Invalid gains, times, capacity or target angles.')
    with args.schedule.open(newline='',encoding='utf-8-sig') as f:
        schedule=[{k:float(r[k]) for k in ('time_s','load_kg')} for r in csv.DictReader(f)]
    if not schedule or schedule[0]['time_s']!=0:raise ValueError('Schedule must start at zero.')
    for i,r in enumerate(schedule):
        if not all(math.isfinite(v) for v in r.values()) or r['load_kg']<0 or r['time_s']>=args.duration:
            raise ValueError('Invalid schedule.')
        if i and r['time_s']<=schedule[i-1]['time_s']:raise ValueError('Times must increase.')
        if abs(r['time_s']/args.dt-round(r['time_s']/args.dt))>1e-7:raise ValueError('Times must align with dt.')
        r.update(kp=args.kp,ki=args.ki)
    import opensim as osim
    osim.Logger.setLevelString('Warn')
    import matplotlib
    if args.no_show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.plt=plt;args.live=not(args.no_show or args.no_live)
    base.DEMO_START_DEG=args.start_deg;base.DEMO_TARGET_DEG=args.target_deg;base.DEMO_CYCLES=args.cycles
    dest=ROOT/('heavy_fixed_PI_results_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]);dest.mkdir()
    args.results_dir=dest;source=base.find_input(ROOT,args.input)
    base.prepare_display(osim,source,args.geometry,False)
    try:base.prepare_display(osim,source,args.geometry,args.live)
    except Exception as exc:base.viewer_error(args,'display setup',exc);args.live=False
    print('FIXED GAINS: Kp=',args.kp,'Ki=',args.ki,'; duration=',args.duration,'s',flush=True)
    print('Results:',dest,flush=True)
    # Both paths in the shared simulator support live display. There is one
    # FIXED trial here: no gain adaptation or scheduled comparison.
    args.fixed_live=True
    rows=variable.run(osim,source,schedule,args,dest,False)
    summaries=variable.metrics(rows,'fixed_PI',schedule,args.duration)
    with (dest/'segment_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    t=[r['time_s'] for r in rows]
    fig,ax=plt.subplots(figsize=(11,3),layout='constrained')
    ax.step(t,[r['load_kg'] for r in rows],where='post',color='#333333',lw=2)
    ax.set(xlabel='Simulation time (s)',ylabel='Dumbbell mass (kg)',title='Physical dumbbell mass during the simulation')
    ax.grid(alpha=.25);fig.savefig(dest/'dumbbell_weight.png',dpi=160)
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True,layout='constrained')
    colors=['#e8f2fa','#fff0cf','#ffe0df','#e8f2fa']
    for j,task in enumerate(schedule):
        end=schedule[j+1]['time_s'] if j+1<len(schedule) else args.duration
        for ax in axes:ax.axvspan(task['time_s'],end,color=colors[j%4],alpha=.5)
        axes[0].text((task['time_s']+end)/2,task['load_kg']+.25,f"{task['load_kg']:g} kg",ha='center')
    axes[0].step(t,[r['load_kg'] for r in rows],where='post',color='black',lw=2);axes[0].set_ylabel('Mass (kg)')
    axes[0].set_ylim(0,max(r['load_kg'] for r in rows)+1)
    axes[1].plot(t,[r['reference_deg'] for r in rows],'k--',label='Target',lw=1.7)
    axes[1].plot(t,[r['angle_deg'] for r in rows],color='#087cab',label='Actual: fixed PI')
    axes[1].set_ylabel('Elbow (deg)');axes[1].legend(loc='upper right')
    axes[2].plot(t,[r['error_deg'] for r in rows],color='#c3532f');axes[2].axhline(0,color='gray',lw=.7)
    axes[2].set_ylabel('Target - actual (deg)')
    axes[3].plot(t,[r['tension_N'] for r in rows],color='#008963',label='Cable tension')
    axes[3].axhline(args.tmax,color='black',ls=':',label='Tension limit');axes[3].set_ylabel('Tension (N)');axes[3].legend(loc='upper right')
    axes[-1].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.grid(alpha=.2)
        for task in schedule[1:]:ax.axvline(task['time_s'],color='gray',ls=':',lw=.8)
    fig.suptitle(f'Changing dumbbell mass with FIXED PI gains\nKp={args.kp:g}, Ki={args.ki:g}; cable limit={args.tmax:g} N',fontsize=15)
    fig.savefig(dest/'fixed_PI_tracking.png',dpi=160);fig.savefig(dest/'fixed_PI_tracking.pdf')
    shutil.copy2(source,dest/'input_model.osim');shutil.copy2(args.schedule,dest/'load_schedule.csv')
    for n in ('run_heavy_load_fixed_PI.py','run_variable_load.py','run_arm26_pi_github.py','add_dumbbell.py'):shutil.copy2(ROOT/n,dest/n)
    settings=dict(completed=True,kp=args.kp,ki=args.ki,duration_s=args.duration,dt_s=args.dt,tmax_N=args.tmax,
                  start_deg=args.start_deg,target_deg=args.target_deg,cycles=args.cycles,leg_time_s=args.rise_time,
                  load_schedule=schedule,grip_body_m=args.grip,method='Fixed PI; no adaptive gains or continual learning')
    (dest/'run_settings.json').write_text(json.dumps(settings,indent=2))
    (dest/'README.txt').write_text(__doc__+'\n\n'+(ROOT/'VARIABLE_LOAD_README.txt').read_text())
    run_id=datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC_')+uuid.uuid4().hex[:8]
    (dest/'upload_manifest.json').write_text(json.dumps(dict(run_id=run_id,files={n:base.file_hash(dest/n) for n in FILES}),indent=2))
    for r in summaries:print(f"Load {r['load_kg']:g} kg: RMSE {r['rmse_deg']:.3f} deg; saturation {r['saturation_pct']:.1f}%",flush=True)
    worker=base.start_github_push(dest,args)
    print('DONE:',dest/'fixed_PI_tracking.png',flush=True)
    if not args.no_show:plt.show()
    if hasattr(args,'display'):args.display.close()
    if worker:worker.join()

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
