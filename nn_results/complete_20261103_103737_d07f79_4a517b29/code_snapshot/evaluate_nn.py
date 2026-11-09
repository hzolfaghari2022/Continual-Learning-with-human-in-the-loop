"""Frozen NN closed-loop tracking, paired with recorded teacher episodes."""
import runtime_setup
import argparse,csv,json,subprocess,sys,time,webbrowser,hashlib
from pathlib import Path
import numpy as np
import simulation_core as c
from dataset_io import resolve_dataset
from numpy_policy import load_policy
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['test','validation'],default='test');p.add_argument('--task',choices=['A','B','all'],default='all')
    p.add_argument('--max-episodes',type=int);p.add_argument('--no-live',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--resume-episodes',action='store_true')
    p.add_argument('--defer-report',action='store_true')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:return c.run_episode(a.worker)
    suite=resolve_dataset(a.dataset or (Path((ROOT/'LATEST_NN_DATASET.txt').read_text().strip()) if (ROOT/'LATEST_NN_DATASET.txt').exists() else None))
    checkpoint=a.checkpoint or Path((ROOT/'LATEST_POLICY.txt').read_text().strip())
    if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
    checkpoint=checkpoint.resolve()
    _,saved=load_policy(checkpoint)
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    candidates=[]
    for path in sorted(suite.glob('*/settings.json')):
        e=c.read_json(path);status=path.parent/'status.json'
        if e['split']!=a.split or (a.task!='all' and e['task']!=a.task):continue
        if not status.exists() or not c.read_json(status).get('accepted_for_imitation',False):continue
        if e.get('controller')!='inverse_dynamics_plus_fixed_PID_acceleration':raise ValueError('Baseline is not corrected teacher data.')
        candidates.append((path.parent,e))
    # Refuse evaluation on training or checkpoint-selection episodes.
    if a.split=='test':
        seen={r['episode_id'] for r in saved['training_records']+saved['validation_records']}
        if any(e['episode_id'] in seen for _,e in candidates):raise ValueError('Test episodes overlap checkpoint fitting/selection records.')
    if a.max_episodes:
        grouped={}
        for item in candidates:grouped.setdefault((item[1]['task'],item[1]['shoulder_deg']),[]).append(item)
        ordered=[]
        while any(grouped.values()):
            for key in sorted(grouped):
                if grouped[key]:ordered.append(grouped[key].pop(0))
        candidates=ordered[:a.max_episodes]
    if not candidates:raise ValueError('No accepted episodes for the chosen split/task.')
    out=(a.output or ROOT/'nn_evaluations'/time.strftime('%Y%m%d_%H%M%S')).resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(baseline,e) in enumerate(candidates,1):
        d=out/e['episode_id'];d.mkdir(exist_ok=True)
        settings=dict(e,neural_checkpoint=str(checkpoint),live_viewer=not a.no_live,save_episode_plots=True)
        reusable=False
        if a.resume_episodes and (d/'evaluation_identity.json').exists() and (d/'status.json').exists():
            identity=c.read_json(d/'evaluation_identity.json');previous=c.read_json(d/'status.json')
            reusable=(identity.get('checkpoint_sha256')==checkpoint_hash and identity.get('baseline_settings_sha256')==hashlib.sha256((baseline/'settings.json').read_bytes()).hexdigest() and previous.get('status')=='completed' and (d/'signals.csv').exists())
        if reusable:
            print(f'[{i}/{len(candidates)}] Reusing completed NN-only rollout: {e["episode_id"]}',flush=True)
        else:
            c.write_json(d/'settings.json',settings)
            print(f'[{i}/{len(candidates)}] NN-only: {e["episode_id"]}',flush=True)
            with (d/'console.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(d)],stdout=log,stderr=subprocess.STDOUT)
                try:proc.wait(timeout=e.get('episode_timeout_s',600))
                except subprocess.TimeoutExpired:
                    proc.terminate();proc.wait(timeout=10)
                    c.write_json(d/'status.json',dict(status='failed',accepted_for_imitation=False,error='NN rollout timed out'))
            c.write_json(d/'evaluation_identity.json',dict(checkpoint_sha256=checkpoint_hash,baseline_settings_sha256=hashlib.sha256((baseline/'settings.json').read_bytes()).hexdigest()))
        r=c.read_json(d/'status.json') if (d/'status.json').exists() else dict(status='failed',accepted_for_imitation=False,error='Worker exited without status')
        teacher=c.read_json(baseline/'status.json');tm=teacher.get('motion_only') or {};nm=r.get('motion_only') or {}
        row=dict(episode_id=e['episode_id'],task=e['task'],split=a.split,shoulder_deg=e['shoulder_deg'],trajectory=e['trajectory'],
            teacher_motion_rmse_deg=tm.get('rmse_deg'),nn_motion_rmse_deg=nm.get('rmse_deg'),
            nn_max_error_deg=(r.get('all_samples') or {}).get('max_abs_error_deg'),nn_status=r['status'],nn_accepted=r.get('accepted_for_imitation',False),
            nn_peak_tension_N=nm.get('peak_tension_N'),nn_saturation_percent=nm.get('clipped_percent'),
            nn_ood_samples=r.get('nn_outside_training_range_samples'),teacher_source=str(baseline))
        rows.append(row);print(json.dumps(row),flush=True)
        if (d/'signals.csv').exists():
            import matplotlib;matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            with np.load(baseline/'training_arrays.npz',allow_pickle=False) as z:
                names=list(z['feature_names']);bt=z['time_s'];bq=np.degrees(z['X_raw'][:,names.index('angle_rad')]);bref=np.degrees(z['X_raw'][:,names.index('reference_rad')]);bf=z['y_force_N'].ravel()
            with (d/'signals.csv').open() as f:signals=list(csv.DictReader(f))
            nt=np.array([float(v['time_s']) for v in signals]);nq=np.array([float(v['angle_deg']) for v in signals]);nf=np.array([float(v['tension_N']) for v in signals])
            fig,axs=plt.subplots(3,1,figsize=(11,10),sharex=True)
            axs[0].plot(bt,bref,'k--',label='Reference');axs[0].plot(bt,bq,label='Recorded compensated PID');axs[0].plot(nt,nq,label='NN only')
            axs[0].legend();axs[0].set_ylabel('Elbow angle (deg)');axs[1].plot(bt,bq-bref,label='Recorded PID error');axs[1].plot(nt,nq-np.interp(nt,bt,bref),label='NN error');axs[1].set_ylabel('Tracking error (deg)');axs[1].legend()
            axs[2].plot(bt,bf,label='Recorded PID force');axs[2].plot(nt,nf,label='NN force');axs[2].set(xlabel='Time (s)',ylabel='Cable force (N)');axs[2].legend()
            for ax in axs:ax.grid(alpha=.2)
            fig.suptitle(e['episode_id']);fig.tight_layout();fig.savefig(d/'comparison.png',dpi=150);plt.close(fig)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    c.write_json(out/'comparison.json',dict(checkpoint=str(checkpoint),split=a.split,rows=rows,
        baseline='Recorded teacher, identical settings; no teacher commands or fallback applied in NN rollout.'))
    html='<html><body><h1>Frozen NN versus recorded teacher</h1><p>NN commands the cable directly. Failed runs are retained.</p>'
    for r in rows:html+=f'<h3>{r["episode_id"]}: accepted={r["nn_accepted"]}</h3><pre>{json.dumps(r,indent=2)}</pre><img width="850" src="{r["episode_id"]}/comparison.png">'
    (out/'VIEW_COMPARISON.html').write_text(html+'</body></html>')
    if out.name=='tracking' and not a.defer_report:
        from reporting import write_report
        print('FULL REPORT:',write_report(out.parent),flush=True)
    print('COMPARISON:',out/'VIEW_COMPARISON.html',flush=True)
    if not a.no_live:webbrowser.open((out/'VIEW_COMPARISON.html').as_uri())
    return 0 if all(r['nn_accepted'] for r in rows) else 2

if __name__=='__main__':raise SystemExit(main())
