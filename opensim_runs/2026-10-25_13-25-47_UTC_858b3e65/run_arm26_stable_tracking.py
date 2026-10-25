"""Force-driven Arm26 tracking: stable adaptive update and matched comparisons.

Known mass is observed by both adaptive learners unless --no-load-input is used.
Importance remains a sensitivity proxy: this is EWC-inspired control, not a
validated demonstration of catastrophic-forgetting prevention.
"""
import argparse,csv,json,math,shutil,sys,traceback,uuid
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import numpy as np
import arm26_ewc_original_core as core

ROOT=Path(__file__).resolve().parent
OriginalLearner=core.ContinualRBFNetworkWithEWC

# Visible export settings: change here or use the command-line switches.
SAVE_VIDEO = True
VIDEO_FPS = 10

class StableLearner(OriginalLearner):
    """Same features for lambda on/off; implicit quadratic-penalty step."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.checkpoints=[]

    def compute_basis(self,q,speed,qref,qdot_ref,error,edot,load_kg,blind=False):
        # Feature selection is independent of whether regularization is enabled.
        saved=self.use_ewc
        self.use_ewc=True
        try:return super().compute_basis(q,speed,qref,qdot_ref,error,edot,load_kg,blind)
        finally:self.use_ewc=saved

    def step_update(self,phi,sliding_error_s,dt,can_learn=True):
        out=self.predict(phi)
        self.segment_phis.append(phi.copy())
        penalty=0.0
        if self.consolidated and self.ewc_lambda>0:
            penalty=.5*self.ewc_lambda*float(np.sum(self.fisher*(self.w-self.anchor)**2))
        if not can_learn or self.lr<=0:return out,penalty
        # Feedback-error task drive; no claim of differentiating through OpenSim.
        derivative=(1-(out/self.tau_limit)**2)*phi
        drive=np.clip(sliding_error_s*derivative,-25.,25.)
        alpha=self.lr*dt
        candidate=self.w+alpha*drive
        if self.consolidated and self.ewc_lambda>0:
            precision=alpha*self.ewc_lambda*self.fisher
            # Exact proximal solution of the local quadratic memory penalty.
            self.w=(candidate+precision*self.anchor)/(1.+precision)
        else:self.w=candidate
        if not np.all(np.isfinite(self.w)):raise FloatingPointError('Nonfinite adaptive weights')
        return out,penalty

    def consolidate_segment(self,seg_idx,load_kg):
        if not self.segment_phis:return
        super().consolidate_segment(seg_idx,load_kg)
        # Single accumulated precision centred on the most recent endpoint.
        # This endpoint is not assumed to be a converged optimum.
        self.anchor=self.w.copy()
        a,b,c,_=self.fisher_snapshots[-1]
        self.fisher_snapshots[-1]=(a,b,c,self.anchor.copy())
        self.checkpoints.append((seg_idx,load_kg,self.w.copy(),self.fisher.copy()))


def save_rows(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot_results(plt,trials,args,schedule,dest):
    colors={'fixed_PI':'#b26633','original_penalty':'#9a6199','matched_no_penalty':'#d65e32','stable_penalty':'#007d9d'}
    labels={'fixed_PI':'Fixed PI baseline','original_penalty':'Uploaded penalty (lambda 95)',
            'matched_no_penalty':'Matched learner, penalty off','stable_penalty':f'Stable update (lambda {args.ewc_lambda:g})'}
    fig,axs=plt.subplots(5,1,sharex=True,figsize=(12,12),layout='constrained',height_ratios=[.55,1,1,1,1])
    rows=trials['stable_penalty'];t=[r['time_s'] for r in rows]
    axs[0].step(t,[r['load_kg'] for r in rows],where='post',color='black');axs[0].set_ylabel('Mass (kg)')
    axs[1].plot(t,[r['reference_deg'] for r in rows],'k--',lw=1.5,label='Reference')
    for name,rs in trials.items():
        for ax,key in zip(axs[1:],['angle_deg','error_deg','tension_N','learned_only_Nm']):
            ax.plot([r['time_s'] for r in rs],[r[key] for r in rs],label=labels[name],color=colors[name],lw=1.25)
    for ax,label in zip(axs[1:],['Angle (deg)','Error (deg)','Tension (N)','Learned torque (N m)']):ax.set_ylabel(label)
    axs[1].legend(ncol=2,fontsize=8,loc='upper right');axs[3].axhline(args.tmax,color='black',lw=1,ls=':',label='Force cap')
    axs[-1].set_xlabel('Simulation time (s)')
    for ax in axs:
        for s in schedule[1:]:ax.axvline(s['time_s'],color='gray',ls=':',lw=.8)
        ax.grid(alpha=.2);ax.set_xlim(0,args.duration)
    fig.suptitle('Same physical load schedule, trajectory, adaptive feedback gains and force cap\nTracking comparison; no claim of proven memory retention',fontsize=13)
    fig.savefig(dest/'tracking_comparison.png',dpi=170);fig.savefig(dest/'tracking_comparison.pdf')
    # Focused figure excludes the intentionally unchanged, oscillatory PI baseline
    # from its zoom only; the full comparison above always includes it.
    zoom,axz=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for name,rs in trials.items():
        if name=='fixed_PI':continue
        for ax,key in zip(axz,['error_deg','tension_N']):
            ax.plot([r['time_s'] for r in rs],[r[key] for r in rs],label=labels[name],color=colors[name],lw=1.3)
    axz[0].set_ylabel('Error (deg)');axz[0].legend(fontsize=9,ncol=1);axz[1].set_ylabel('Tension (N)');axz[1].set_xlabel('Time (s)')
    for ax in axz:ax.set_xlim(32.5,48.5);ax.grid(alpha=.2)
    zoom.suptitle('Heavy-load segment: 5 kg | matched adaptive feedback and 240 N cap' if args.tmax==240 else f'Heavy-load segment: 5 kg | cap {args.tmax:g} N')
    zoom.savefig(dest/'heavy_load_zoom.png',dpi=170);zoom.savefig(dest/'heavy_load_zoom.pdf')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'arm26_exosuit.osim')
    p.add_argument('--geometry',default=None)
    p.add_argument('--ewc-lambda',type=float,default=5.)
    p.add_argument('--ewc-gamma',type=float,default=.95)
    p.add_argument('--tmax',type=float,default=240.)
    p.add_argument('--nn-lr',type=float,default=.95)
    p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--no-load-input',action='store_true')
    p.add_argument('--no-show',action='store_true')
    p.add_argument('--no-live',action='store_true')
    p.add_argument('--no-video',action='store_true',help='Skip recorded-state movie export.')
    p.add_argument('--video-fps',type=int,default=VIDEO_FPS,help='Recorded-state video frame rate.')
    p.add_argument('--no-push',action='store_true')
    p.add_argument('--export-existing',type=Path,help='Save figures, movie and report for a completed run; no new simulation.')
    p.add_argument('--push-results',type=Path,default=None,help='Retry upload of an existing completed folder, without rerunning physics.')
    p.add_argument('--skip-original',action='store_true',help='Skip the old update diagnostic; still runs matched no-penalty and fixed PI.')
    args=p.parse_args()
    core.base.GITHUB_RUNNER=str(Path(__file__).resolve())
    if args.export_existing is not None:
        from export_simulation_media import recover_exports
        recover_exports(args.export_existing,geometry=args.geometry,make_video=SAVE_VIDEO and not args.no_video,fps=args.video_fps,show=not args.no_show)
        return
    if args.push_results is not None:
        folder=args.push_results.expanduser().resolve()
        manifest=json.loads((folder/'upload_manifest.json').read_text())
        core.base.GITHUB_FILES=tuple(manifest['files'])
        if not core.base.push_completed_run(folder):raise RuntimeError('GitHub retry failed; see github_push_status.txt.')
        return
    if args.video_fps<1 or args.video_fps>60:raise ValueError('Video FPS must be 1 to 60')
    if args.dt<=0 or args.tmax<=0 or args.nn_lr<0 or args.ewc_lambda<0 or not 0<=args.ewc_gamma<=1:raise ValueError('Invalid numeric option')
    import opensim as osim
    osim.Logger.setLevelString('Warn')
    import matplotlib
    if args.no_show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,value in dict(kp=30.,ki=15.,kv=0.,kp_5kg=95.,ki_5kg=55.,kv_5kg=2.5,
        start_deg=20.,target_deg=80.,rise_time=4.,cycles=8,grip=core.DEFAULT_GRIP,
        kp_nn=38.,kd_nn=7.,kd_fel=8.,nn_tau_max=45.,duration=65.5).items():setattr(args,key,value)
    args.live=not(args.no_show or args.no_live);args.plt=plt
    source=args.input.expanduser().resolve()
    if not source.is_file():raise FileNotFoundError(source)
    core.base.DEMO_START_DEG=20.;core.base.DEMO_TARGET_DEG=80.;core.base.DEMO_CYCLES=8
    schedule=[dict(time_s=t,load_kg=w) for t,w in [(0,.5),(16.5,2.),(32.5,5.),(48.5,.5)]]
    run_id=datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC_')+uuid.uuid4().hex[:8]
    output_root=ROOT/'simulation_outputs';output_root.mkdir(exist_ok=True)
    dest=output_root/('stable_tracking_'+run_id);dest.mkdir();args.results_dir=dest
    (ROOT/'LATEST_RESULTS_PATH.txt').write_text(str(dest.resolve())+'\n')
    print('\nARM26 MEDIA EXPORT V3 | Figures, movie and report save automatically.',flush=True)
    print('SAVE FOLDER: '+str(dest.resolve()),flush=True)
    core.base.prepare_display(osim,source,args.geometry,args.live)
    display=None
    if args.live:
        try:display=core.PersistentLiveDisplay(osim,source,args,plt)
        except Exception as exc:core.base.viewer_error(args,'stable display',exc)
    trials={};metrics=[];summaries=[]
    configs=[('stable_penalty',StableLearner,args.ewc_lambda),('matched_no_penalty',StableLearner,0.)]
    if not args.skip_original:configs.append(('original_penalty',OriginalLearner,95.))
    configs.append(('fixed_PI',OriginalLearner,0.))
    core.base.AUTO_PUSH_TO_GITHUB=True
    for label,learner_type,lam in configs:
        core.ContinualRBFNetworkWithEWC=learner_type
        trial_args=SimpleNamespace(**(vars(args)|{'ewc_lambda':lam}))
        folder=dest/label;folder.mkdir();mode='fixed_PI' if label=='fixed_PI' else 'ewc_NN'
        print(f'\nCASE {label}: lambda={lam:g}; LR={args.nn_lr:g}; PI/PD terms unchanged.',flush=True)
        rows,status,reason,learner=core.run_continual_trial(osim,source,schedule,trial_args,folder,mode,
            display=display if label=='stable_penalty' else None)
        for row in rows:
            _,vr=core.ref_and_vel(row['time_s'],args.rise_time)
            pd=0. if mode=='fixed_PI' else args.kp_nn*math.radians(row['error_deg'])+args.kd_nn*(vr-row['velocity_rad_s'])
            row['handcoded_PD_Nm']=pd;row['learned_only_Nm']=row['NN_Nm']-pd
            row['Ki_effective']=args.ki if mode=='fixed_PI' else .35*args.ki
        save_rows(dest/f'{label}_data.csv',rows)
        if learner is not None:
            np.savez(folder/'learner_final.npz',weights=learner.w,importance_proxy=learner.fisher,anchor=learner.anchor)
            if getattr(learner,'checkpoints',None):
                np.savez(folder/'task_checkpoints.npz',segments=np.array([x[0] for x in learner.checkpoints]),
                    masses=np.array([x[1] for x in learner.checkpoints]),weights=np.stack([x[2] for x in learner.checkpoints]),
                    importance_proxy=np.stack([x[3] for x in learner.checkpoints]))
        trials[label]=rows
        per=core.variable.metrics(rows,label,schedule,args.duration);metrics.extend(per)
        complete=status=='completed' and abs(rows[-1]['time_s']-args.duration)<1e-6
        summary=dict(controller=label,status=status,completed=complete,failure=reason,samples=len(rows),
            overall_rmse_deg=math.sqrt(sum(r['error_deg']**2 for r in rows)/len(rows)),lambda_value=lam)
        summaries.append(summary);print('SUMMARY',summary,flush=True)
        (dest/'trial_status.json').write_text(json.dumps(summaries,indent=2))
    if display:display.finish()
    save_rows(dest/'segment_metrics.csv',metrics)
    plot_results(plt,trials,args,schedule,dest)
    shutil.copy2(source,dest/'input_model.osim')
    code_files=['run_arm26_stable_tracking.py','arm26_ewc_original_core.py','run_variable_load.py','run_arm26_pi_github.py','add_dumbbell.py','export_simulation_media.py','report_template.tex','control_block_diagram.tex','RUN_THIS.py','SAVE_EXISTING_RUN.py']
    for name in code_files:shutil.copy2(ROOT/name,dest/name)
    settings={k:v for k,v in vars(args).items() if isinstance(v,(str,int,float,bool,list,tuple)) or v is None}
    settings.update(run_id=run_id,completed=all(s['completed'] for s in summaries),load_schedule=schedule,
        method='Feedback-error residual control with proximal sensitivity-weighted regularization',
        source_model=source.name,adaptive_Ki_effective=.35*args.ki,known_mass_input=not args.no_load_input,
        limitation='Tracking-tuned on this sequence. Frozen-weight retention and held-out tasks are not validated.',
        opensim_version=osim.GetVersion(),summaries=summaries)
    (dest/'run_settings.json').write_text(json.dumps(settings,indent=2))
    (dest/'README.txt').write_text(__doc__+'\nSee run_settings.json and trial_status.json. Any incomplete cases remain labelled; no traces are filled in.\n')
    # All exports finish before the GitHub snapshot is assembled.
    from export_simulation_media import export_all
    export_all(dest,geometry=args.geometry,make_video=SAVE_VIDEO and not args.no_video,fps=args.video_fps)
    # Reuse existing authenticated uploader, with complete nested-file coverage.
    core.base.GITHUB_FILES=tuple(sorted(p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()))
    (dest/'upload_manifest.json').write_text(json.dumps(dict(run_id=run_id,files={n:core.base.file_hash(dest/n) for n in core.base.GITHUB_FILES}),indent=2))
    worker=core.base.start_github_push(dest,args)
    print('\nSAVE FOLDER: '+str(dest.resolve()),flush=True)
    print('Open export_status.json for exact saved paths and any export errors.',flush=True)
    if not args.no_show:
        from export_simulation_media import show_all_figures
        show_all_figures(dest)
    if worker:worker.join()
    if not settings['completed']:raise RuntimeError('One or more trials incomplete; inspect saved trial_status.json.')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('ERROR:',exc,file=sys.stderr);traceback.print_exc();sys.exit(1)
