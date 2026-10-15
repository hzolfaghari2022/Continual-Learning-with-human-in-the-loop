"""Compare constant loads with the same PI gains, motion and initial states.

Each load starts a fresh independent OpenSim trial. Default figure overlays
0.5/2/5 kg at the original experiment's 120 N cap. A separate 240 N capacity
check tests whether force saturation explains the failure. No gain adaptation.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import uuid
from plot_combined_scenarios import plot_combined
import run_variable_load as variable
import run_arm26_pi_github as base
from add_dumbbell import DEFAULT_GRIP

ROOT=Path(__file__).resolve().parent

def plot_primary(plt,records,loads,capacity,kp,ki,reference_times,refs,dest):
    colors=['#087cab','#d87b22','#ac3d52','#009d69','#7360a5']
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    axes[0].plot(reference_times,refs,'k--',lw=1.8,label='Target')
    for cap,load,rows,status in records:
        if cap!=capacity or not rows:continue
        label=f'{load:g} kg'+(' (FAILED)' if status!='completed' else '')
        for ax,key in zip(axes,['angle_deg','error_deg','tension_N']):
            ax.plot([r['time_s'] for r in rows],[r[key] for r in rows],label=label,
                    color=colors[loads.index(load)%len(colors)],lw=1.4)
    axes[0].set_ylabel('Elbow angle (deg)');axes[1].set_ylabel('Target - actual (deg)')
    axes[2].set_ylabel('Cable tension (N)');axes[2].set_xlabel('Simulation time (s)')
    axes[1].axhline(0,color='gray',lw=.7)
    axes[2].axhline(capacity,color='black',ls=':',label=f'{capacity:g} N limit')
    for ax in axes:ax.grid(alpha=.2);ax.legend(loc='upper right',ncol=4,fontsize=9)
    fig.suptitle(f'Different loads, SAME fixed PI gains and target\nKp={kp:g}, Ki={ki:g}; cable limit={capacity:g} N; independent trials',fontsize=15)
    fig.savefig(dest/'fixed_PI_load_overlay.png',dpi=170)
    fig.savefig(dest/'fixed_PI_load_overlay.pdf')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--loads',type=float,nargs='+',default=[.5,2,5])
    p.add_argument('--tmax',type=float,default=120)
    p.add_argument('--capacity-check',type=float,default=240,help='Secondary cap, N. Set 0 to skip.')
    p.add_argument('--kp',type=float,default=30);p.add_argument('--ki',type=float,default=15)
    p.add_argument('--cycles',type=int,default=8);p.add_argument('--rise-time',type=float,default=4)
    p.add_argument('--start-deg',type=float,default=20);p.add_argument('--target-deg',type=float,default=80)
    p.add_argument('--dt',type=float,default=.01);p.add_argument('--grip',type=float,nargs=3,default=DEFAULT_GRIP)
    p.add_argument('--input',default=None);p.add_argument('--geometry',default=None)
    p.add_argument('--no-show',action='store_true');p.add_argument('--no-live',action='store_true')
    p.add_argument('--no-push',action='store_true');p.add_argument('--push-results',type=Path,default=None)
    args=p.parse_args();base.GITHUB_RUNNER=__file__
    if args.push_results:
        manifest=json.loads((args.push_results/'upload_manifest.json').read_text())
        for name in manifest['files']:
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or '.git' in path.parts:
                raise ValueError('Invalid snapshot path.')
        base.GITHUB_FILES=tuple(manifest['files'])
        if not base.push_completed_run(args.push_results.resolve()):raise SystemExit(2)
        return
    args.duration=.5+2*args.cycles*args.rise_time+1
    numeric=[args.kp,args.ki,args.tmax,args.capacity_check,args.dt,args.rise_time,args.start_deg,args.target_deg,*args.loads,*args.grip]
    if not all(math.isfinite(v) for v in numeric):raise ValueError('Parameters must be finite.')
    if min(args.ki,args.dt,args.rise_time,args.tmax)<=0 or args.kp<0 or min(args.loads)<0 or args.capacity_check<0 or args.cycles<1:
        raise ValueError('Invalid gains, mass, capacity or timing.')
    if not 0<=args.start_deg<args.target_deg<=90:raise ValueError('Angles must lie within 0-90 degrees.')
    if len(set(args.loads))!=len(args.loads):raise ValueError('Specify each load once.')
    import opensim as osim
    osim.Logger.setLevelString('Warn')
    import matplotlib
    if args.no_show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.plt=plt;args.live=False;args.fixed_live=True
    base.DEMO_START_DEG=args.start_deg;base.DEMO_TARGET_DEG=args.target_deg;base.DEMO_CYCLES=args.cycles
    source=base.find_input(ROOT,args.input)
    base.prepare_display(osim,source,args.geometry,False)
    dest=ROOT/('load_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]);dest.mkdir()
    caps=[args.tmax]
    if args.capacity_check and args.capacity_check!=args.tmax:caps.append(args.capacity_check)
    reference_times=[i*args.dt for i in range(round(args.duration/args.dt)+1)]
    refs=[math.degrees(base.reference(t,args.rise_time)) for t in reference_times]
    print('Computing constant-load cases first; the changing-load case opens the live viewer afterward.',flush=True)
    records=[];summaries=[]
    for capacity in caps:
        for load in args.loads:
            args.tmax=capacity
            case=f'load_{load:g}kg_cap_{capacity:g}N'
            folder=dest/case;folder.mkdir();args.results_dir=folder
            # Only the changing-load run opens a live viewer.
            args.live=False
            if args.live:
                try:base.prepare_display(osim,source,args.geometry,True)
                except Exception as exc:base.viewer_error(args,'display setup',exc);args.live=False
            schedule=[dict(time_s=0,load_kg=load,kp=args.kp,ki=args.ki)]
            print(f'CASE {case}: identical Kp={args.kp:g}, Ki={args.ki:g}; {args.duration:g}s',flush=True)
            args.trial_rows=[]
            try:
                rows=variable.run(osim,source,schedule,args,folder,False)
                status='completed';failure=''
            except Exception as exc:
                rows=args.trial_rows;status='FAILED';failure=f'{type(exc).__name__}: {exc}'
                (folder/'failure.txt').write_text(failure)
                print(f'{case}: {failure}; partial data retained; continuing.',flush=True)
                if rows:
                    with (folder/'partial_data.csv').open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                if hasattr(args,'display'):
                    args.display.close();del args.display
            metric=dict(case=case,load_kg=load,tmax_N=capacity,kp=args.kp,ki=args.ki,status=status,
                        stop_time_s=rows[-1]['time_s'] if rows else 0,rmse_deg='',late_rmse_deg='',peak_abs_error_deg='',
                        upper_saturation_pct='',lower_saturation_pct='',peak_tension_N='',failure=failure)
            if rows:
                late=[r for r in rows if r['time_s']>=args.duration/2]
                metric.update(rmse_deg=math.sqrt(sum(r['error_deg']**2 for r in rows)/len(rows)),
                              peak_abs_error_deg=max(abs(r['error_deg']) for r in rows),
                              upper_saturation_pct=100*sum(r['torque_raw_Nm']>=capacity*r['moment_arm_m']-1e-8 for r in rows)/len(rows),
                              lower_saturation_pct=100*sum(r['torque_raw_Nm']<=0 for r in rows)/len(rows),
                              peak_tension_N=max(r['tension_N'] for r in rows))
                if late:metric['late_rmse_deg']=math.sqrt(sum(r['error_deg']**2 for r in late)/len(late))
            records.append((capacity,load,rows,status));summaries.append(metric)
            print(f'{case}: {status}; RMSE={metric["rmse_deg"]}; upper saturation={metric["upper_saturation_pct"]}%',flush=True)
    # One additional force-driven run supplies the changing-load trace.
    levels=[args.loads[0],args.loads[len(args.loads)//2],args.loads[-1],args.loads[0]]
    event_times=[0]+[round((.5+2*args.cycles*args.rise_time*j/4)/args.dt)*args.dt for j in (1,2,3)]
    schedule=[dict(time_s=t,load_kg=m,kp=args.kp,ki=args.ki) for t,m in zip(event_times,levels)]
    if any(b['time_s']<=a['time_s'] for a,b in zip(schedule,schedule[1:])):
        raise ValueError('Time resolution is too coarse for four load intervals.')
    folder=dest/'changing_load';folder.mkdir();args.results_dir=folder;args.tmax=caps[0]
    args.live=not(args.no_show or args.no_live)
    if args.live:
        try:base.prepare_display(osim,source,args.geometry,True)
        except Exception as exc:base.viewer_error(args,'display setup',exc);args.live=False
    args.trial_rows=[]
    try:
        changing=variable.run(osim,source,schedule,args,folder,False)
        changing_ok=True
    except Exception as exc:
        changing=args.trial_rows;changing_ok=False
        (folder/'failure.txt').write_text(f'{type(exc).__name__}: {exc}')
        if changing:
            with (folder/'partial_data.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(changing[0]));w.writeheader();w.writerows(changing)
        if hasattr(args,'display'):args.display.close();del args.display
    with (folder/'load_schedule.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(schedule[0]));w.writeheader();w.writerows(schedule)
    if changing:
        metrics=variable.metrics(changing,'fixed_PI',schedule,changing[-1]['time_s'])
        with (folder/'segment_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    combined=[(f'Constant {m:g} kg',rows,False,status=='completed') for cap,m,rows,status in records if cap==caps[0]]
    combined.append(('Changing load',changing,True,changing_ok))
    plot_combined(plt,combined,args.kp,args.ki,caps[0],reference_times,refs,schedule,dest)
    colors=['#087cab','#d87b22','#ac3d52','#009d69','#7360a5']
    fig,axes=plt.subplots(3,len(caps),figsize=(14 if len(caps)>1 else 12,10),sharex=True,squeeze=False,layout='constrained')
    for col,capacity in enumerate(caps):
        a=axes[:,col]
        a[0].plot(reference_times,refs,'k--',lw=1.5,label='Target')
        for capacity2,load,rows,status in records:
            if capacity2!=capacity or not rows:continue
            color=colors[args.loads.index(load)%len(colors)];t=[r['time_s'] for r in rows]
            label=f'{load:g} kg'+(' (FAILED)' if status!='completed' else '')
            for ax,key in zip(a,['angle_deg','error_deg','tension_N']):
                ax.plot(t,[r[key] for r in rows],color=color,label=label,lw=1.2)
        a[0].set_title(f'Cable limit: {capacity:g} N'+(' — primary comparison' if col==0 else ' — capacity check'))
        a[1].axhline(0,color='gray',lw=.7);a[2].axhline(capacity,color='black',ls=':',label='Limit')
        a[0].set_ylabel('Elbow angle (deg)');a[1].set_ylabel('Target - actual (deg)');a[2].set_ylabel('Tension (N)')
        for ax in a:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right',ncol=2)
        a[-1].set_xlabel('Simulation time (s)')
    # Common scales make visual comparison across capacities honest.
    for row in range(3):
        lo=min(axes[row,col].get_ylim()[0] for col in range(len(caps)))
        hi=max(axes[row,col].get_ylim()[1] for col in range(len(caps)))
        for col in range(len(caps)):axes[row,col].set_ylim(lo,hi)
    fig.suptitle(f'Same PI gains and target; different physical loads\nKp={args.kp:g}, Ki={args.ki:g}; independent trials with the same initial state',fontsize=15)
    fig.savefig(dest/'all_loads_comparison.png',dpi=170);fig.savefig(dest/'all_loads_comparison.pdf')
    plot_primary(plt,records,args.loads,caps[0],args.kp,args.ki,reference_times,refs,dest)
    with (dest/'comparison_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    shutil.copy2(source,dest/'input_model.osim')
    for name in ['compare_fixed_PI_loads.py','plot_combined_scenarios.py','run_variable_load.py','run_arm26_pi_github.py','add_dumbbell.py']:
        shutil.copy2(ROOT/name,dest/name)
    settings=dict(kp=args.kp,ki=args.ki,loads_kg=args.loads,tension_caps_N=caps,duration_s=args.duration,dt_s=args.dt,
                  start_deg=args.start_deg,target_deg=args.target_deg,cycles=args.cycles,leg_time_s=args.rise_time,
                  grip_body_m=args.grip,changing_load_schedule=schedule,changing_load_completed=changing_ok,
                  opensim_version=osim.GetVersion(),completed=changing_ok and all(r['status']=='completed' for r in summaries))
    (dest/'run_settings.json').write_text(json.dumps(settings,indent=2))
    (dest/'README.txt').write_text(__doc__+'\n'+(ROOT/'LOAD_COMPARISON_README.txt').read_text())
    # Include complete and failed cases alike; never publish only successful cases.
    files=tuple(f.relative_to(dest).as_posix() for f in sorted(dest.rglob('*')) if f.is_file())
    base.GITHUB_FILES=files
    run_id=datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC_')+uuid.uuid4().hex[:8]
    (dest/'upload_manifest.json').write_text(json.dumps(dict(run_id=run_id,files={n:base.file_hash(dest/n) for n in files}),indent=2))
    worker=base.start_github_push(dest,args)
    print('DONE — combined scenario figure:',dest/'all_scenarios_fixed_PI.png',flush=True)
    if not args.no_show:plt.show()
    if hasattr(args,'display'):args.display.close()
    if worker:worker.join()

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
