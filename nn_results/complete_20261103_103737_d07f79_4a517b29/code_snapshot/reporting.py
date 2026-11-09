"""Saved scientific figures and a printable HTML report; no controller changes."""
import runtime_setup
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':12,'axes.labelsize':13,'legend.fontsize':10})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def training_figures(out, history, splits, metrics):
    out=Path(out)
    epochs=[r['epoch'] for r in history]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(epochs,[r['training_force_rmse_N'] for r in history],label='Train')
    ax.plot(epochs,[r['validation_force_rmse_N'] for r in history],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Force RMSE (N)',title='Learning curves (same metric on both splits)');ax.legend();ax.grid(alpha=.25)
    save(fig,out/'learning_curve.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    names=list(splits)
    axes[0].bar(names,[len(v[3]) for v in splits.values()]);axes[0].set_ylabel('Accepted episodes')
    axes[1].bar(names,[len(v[0]) for v in splits.values()]);axes[1].set_ylabel('Time samples')
    fig.suptitle('Data allocation: whole episodes kept separate');save(fig,out/'data_split.png')
    manifest=[]
    for name,(x,y,p,records,ids) in splits.items():
        manifest.extend(records)
        y=y.ravel()*650;p=p.ravel()*650;error=p-y
        # Deterministic subsampling is for plots only; metrics use every sample.
        ix=np.linspace(0,len(y)-1,min(12000,len(y))).astype(int)
        fig,ax=plt.subplots(figsize=(6,6))
        ax.scatter(y[ix],p[ix],s=3,alpha=.25)
        limits=[float(min(y.min(),p.min())),float(max(y.max(),p.max()))]
        ax.plot(limits,limits,'k--');ax.set(xlabel='Recorded PID force (N)',ylabel='Raw NN prediction (N)',title=f'{name}: predicted versus recorded force');ax.grid(alpha=.25)
        save(fig,out/f'{name}_parity.png')
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].hist(error,bins=80);axes[0].set(xlabel='NN − PID force (N)',ylabel='Samples')
        axes[1].scatter(y[ix],error[ix],s=3,alpha=.25);axes[1].axhline(0,color='k',ls='--');axes[1].set(xlabel='PID force (N)',ylabel='NN − PID force (N)')
        fig.suptitle(f'{name}: force residuals');save(fig,out/f'{name}_residuals.png')
        per=[]
        for eid in dict.fromkeys(ids.tolist()):
            mask=ids==eid;e=error[mask]
            per.append({'episode_id':eid,'split':name,'samples':int(mask.sum()),'rmse_N':float(np.sqrt(np.mean(e**2))),'max_error_N':float(abs(e).max())})
        with (out/f'{name}_episode_force_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
        fig,ax=plt.subplots(figsize=(max(9,len(per)*.25),5))
        ax.bar(range(len(per)),[r['rmse_N'] for r in per]);ax.set_xticks(range(len(per)),[r['episode_id'] for r in per],rotation=90,fontsize=7);ax.set_ylabel('Force RMSE (N)');ax.set_title(f'{name}: every accepted episode');save(fig,out/f'{name}_episode_errors.png')
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))


def table(rows, columns):
    return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c,'')))+'</td>' for c in columns)+'</tr>' for r in rows)+'</table>'


def write_report(out):
    out=Path(out)
    esc=html.escape
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>Arm26 neural controller report</title><style>body{max-width:1100px;margin:35px auto;font:17px Georgia;line-height:1.5;padding:20px}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #aaa;padding:7px}pre{white-space:pre-wrap}h1,h2{color:#173958}@media print{h2{break-before:auto}img{max-height:650px;object-fit:contain}tr,img{break-inside:avoid}}</style></head><body><h1>Arm26 neural controller report</h1>']
    if (out/'run_info.json').exists():
        info=json.loads((out/'run_info.json').read_text());parts+=['<h2>Run configuration</h2><pre>'+esc(json.dumps(info,indent=2))+'</pre>']
    parts+=['<h2>Method and data separation</h2><p>The neural network predicts cable tension from arm state, reference motion, load, shoulder angle, previous force, error integral, and muscle states. Supervised targets are the recorded compensated PID forces. Only accepted, compatible teacher episodes are included. Original train/validation/test assignments are preserved. No time samples from an episode are assigned to another split. Validation selects the checkpoint; test force data are evaluated only after selection. Scaling is fixed using physical units, without fitting on validation or test data.</p><p>The 21–64–64–1 tanh network commands cable force directly during OpenSim evaluation, clipped to 0–650 N. The rollout has no PID force fallback. Good force prediction alone does not establish good closed-loop tracking or superiority to the teacher.</p>']
    if (out/'split_manifest.json').exists():
        records=json.loads((out/'split_manifest.json').read_text());counts=[]
        for split in ['train','validation','test']:
            rr=[r for r in records if r['split']==split]
            counts.append(dict(split=split,episodes=len(rr),samples=sum(r['samples'] for r in rr)))
        parts+=['<h2>Actual data used</h2>'+table(counts,['split','episodes','samples']),'<p>See split_manifest.json for exact episode IDs and dataset file hashes. Counts reflect accepted episodes; rejection can change proportions from the planned 70/15/15 allocation.</p>']
    if (out/'force_metrics.json').exists():
        metrics=json.loads((out/'force_metrics.json').read_text());rows=[]
        for name in ['training','validation','test']:
            if name in metrics:rows.append(dict(split=name,**metrics[name]))
        parts+=['<h2>Force imitation results</h2>'+table(rows,['split','rmse_N','mae_N','max_error_N','r2'])]
        if 'test_unavailable' in metrics:parts+=['<p>Test unavailable: '+esc(metrics['test_unavailable'])+'</p>']
    for path in sorted(out.glob('*.png')):
        parts+=['<h3>'+esc(path.stem.replace('_',' '))+'</h3><img src="'+esc(path.name)+'">']
    tracking=out/'tracking'/'comparison.json'
    if tracking.exists():
        data=json.loads(tracking.read_text());rows=data['rows'];ok=sum(bool(r['nn_accepted']) for r in rows)
        parts+=['<h2>Held-out OpenSim tracking</h2><p>'+str(ok)+' / '+str(len(rows))+' evaluated episodes passed the tracking criteria. Motion RMSE and full-episode maximum error are separate metrics. Required acceptance: completed rollout, motion RMSE ≤ 2°, full-episode maximum error ≤ 5°, plus the simulator’s validity checks. These results cover the listed test episodes only. OOD samples count states outside at least one training feature range; this is a diagnostic, not proof of failure or safety.</p>',table(rows,['episode_id','teacher_motion_rmse_deg','nn_motion_rmse_deg','nn_max_error_deg','nn_status','nn_accepted','nn_saturation_percent','nn_ood_samples'])]
        valid=[r for r in rows if r['nn_motion_rmse_deg'] is not None and r['teacher_motion_rmse_deg'] is not None]
        if valid:
            fig,ax=plt.subplots(figsize=(max(10,len(valid)*.4),6));x=np.arange(len(valid));ax.bar(x-.2,[r['teacher_motion_rmse_deg'] for r in valid],.4,label='Recorded PID');ax.bar(x+.2,[r['nn_motion_rmse_deg'] for r in valid],.4,label='NN');ax.set_xticks(x,[r['episode_id'] for r in valid],rotation=90,fontsize=7);ax.set_ylabel('Motion tracking RMSE (deg)');ax.legend();save(fig,out/'tracking_summary.png')
            parts+=['<img src="tracking_summary.png">']
        for r in rows:
            rel=Path('tracking')/r['episode_id']/'comparison.png'
            parts+=['<h3>'+esc(r['episode_id'])+'</h3>']
            if (out/rel).exists():parts+=['<img src="'+rel.as_posix()+'">']
            else:parts+=['<p>Rollout plot unavailable. Inspect the episode status.json and console.log.</p>']
    else:parts+=['<h2>Tracking status</h2><p>No closed-loop evaluation is included yet. Training completion alone is not proof of successful tracking.</p>']
    if (out/'pipeline_status.json').exists():parts+=['<h2>Execution status</h2><pre>'+esc((out/'pipeline_status.json').read_text())+'</pre>']
    parts+=['<p>This report can be printed or saved as PDF from a browser. Keep the report beside its PNG files and tracking folder.</p></body></html>']
    target=out/'REPORT.html';target.write_text(''.join(parts),encoding='utf-8')
    # Export after all figures are written; no torch imports or simulation reruns.
    try:
        from generate_overleaf import build_report
        print('OVERLEAF REPORT:',build_report(out),flush=True)
        (out/'overleaf_export_error.txt').unlink(missing_ok=True)
    except Exception as error:
        # A reporting failure must remain visible without hiding the HTML results.
        (out/'overleaf_export_error.txt').write_text(str(error),encoding='utf-8')
        print('Overleaf export failed:',error,flush=True)
    return target
