"""Export an existing Arm26 run to Overleaf; no fitting or physics reruns."""
import runtime_setup
import argparse,csv,json,shutil,zipfile,math,re,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tex(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in str(value))


def number(v):
    if v is None:return '--'
    return f'{v:.5g}' if isinstance(v,(float,int)) else tex(v)


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def caption(path):
    name=path.stem
    if name=='learning_curve':return 'Training and validation force RMSE per epoch. Validation selects the saved checkpoint; the test set is not used for selection.'
    if name=='data_split':return 'Actual accepted episode and sample counts. Whole episodes remain in their original split.'
    if 'parity' in name:return 'Predicted versus recorded compensated-PID force. Agreement with the diagonal indicates accurate force imitation; it does not by itself establish closed-loop tracking.'
    if 'residual' in name:return 'Force residuals (NN minus recorded PID). The distribution and force-dependent scatter expose bias and large errors.'
    if 'episode_errors' in name:return 'Force RMSE for every included episode in this split. Large episode-specific errors remain visible.'
    if name=='comparison':return 'Held-out OpenSim rollout: reference, recorded PID, and NN elbow motion; tracking errors; and cable forces. The NN supplies the cable command without a PID fallback.'
    if name=='tracking_summary' or name=='tracking_evidence':return 'Tracking accuracy across every evaluated episode. Motion RMSE excludes the configured holds; maximum error covers the complete recorded episode.'
    if name=='duration_audit':return 'Recorded simulation duration and computation time for each episode. Simulation time and wall-clock time are different quantities.'
    return 'Saved simulation diagnostics. Quantities and units are identified on the axes; failed runs are retained.'


def write_fig(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)


def build_report(run):
    run=Path(run).resolve()
    if not (run/'history.csv').exists():raise FileNotFoundError('history.csv missing: choose the complete recovered/trained run folder.')
    info=read(run/'run_info.json',{});metrics=read(run/'force_metrics.json',{});manifest=read(run/'split_manifest.json',[])
    comparison=read(run/'tracking'/'comparison.json',{});rows=comparison.get('rows',[])
    status=read(run/'pipeline_status.json',{})
    with (run/'history.csv').open(newline='') as f:history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    # Rebuild saved training figures, using arrays already exported by the recovery workflow.
    splits={}
    for split in ('train','validation','test'):
        path=run/(split+'_plot_data.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                splits[split]=(np.empty((len(z['y']),0)),z['y'],z['p'],json.loads(str(z['records_json'])),z['ids'])
    if splits:
        from reporting import training_figures
        training_figures(run,history,splits,metrics)
        manifest=read(run/'split_manifest.json',[])
    # Independently verify episode separation before making an evidence claim.
    groups={s:{r['episode_id'] for r in manifest if r['split']==s} for s in ('train','validation','test')}
    if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Split manifest contains overlapping episode IDs.')
    export=run/'overleaf_report';export.mkdir(exist_ok=True)
    figures=export/'overleaf_figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':11})
    if rows:
        valid=[r for r in rows if r.get('nn_motion_rmse_deg') is not None and r.get('teacher_motion_rmse_deg') is not None]
        if valid:
            fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True);x=np.arange(len(rows))
            for offset,key,label in [(-.2,'teacher_motion_rmse_deg','Recorded PID'),(.2,'nn_motion_rmse_deg','NN')]:
                axes[0].bar(x+offset,[r.get(key) if r.get(key) is not None else np.nan for r in rows],.4,label=label)
            axes[0].set_ylabel('Motion RMSE (deg)');axes[0].legend()
            axes[1].bar(x,[r.get('nn_max_error_deg') if r.get('nn_max_error_deg') is not None else np.nan for r in rows]);axes[1].set_ylabel('NN full maximum error (deg)')
            axes[1].set_xticks(x,[r['episode_id'] for r in rows],rotation=90,fontsize=7)
            for ax in axes:ax.grid(axis='y',alpha=.2)
            write_fig(fig,run/'tracking_evidence.png')
    audit=[]
    for r in rows:
        folder=run/'tracking'/r['episode_id'];s=read(folder/'status.json',{});cfg=read(folder/'settings.json',{})
        signal=folder/'signals.csv';times=[]
        if signal.exists():
            with signal.open(newline='') as f:times=[float(v['time_s']) for v in csv.DictReader(f)]
        dif=np.diff(times)
        audit.append(dict(episode_id=r['episode_id'],samples=len(times),first_time_s=times[0] if times else None,last_time_s=times[-1] if times else None,
            elapsed_simulation_s=times[-1]-times[0] if len(times)>1 else None,median_sample_interval_s=float(np.median(dif)) if len(dif) else None,
            configured_dt_s=cfg.get('dt_s'),wall_seconds=s.get('wall_seconds'),monotonic_time=bool(len(times)>1 and np.all(dif>0)),status=s.get('status','missing')))
    if audit:
        with (export/'sampling_audit.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
        fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(audit))
        ax.bar(x-.2,[r['elapsed_simulation_s'] if r['elapsed_simulation_s'] is not None else np.nan for r in audit],.4,label='Recorded simulation time')
        ax.bar(x+.2,[r['wall_seconds'] if r['wall_seconds'] is not None else np.nan for r in audit],.4,label='Computation time')
        ax.set_xticks(x,[str(i+1) for i in x]);ax.set(xlabel='Test episode index (same order as metrics table)',ylabel='Time (s)');ax.legend();write_fig(fig,run/'duration_audit.png')
    # Copy every saved PNG under the results root and tracking, including every episode.
    sources=sorted(run.glob('*.png'))+sorted((run/'tracking').rglob('*.png'))
    included=[]
    for i,path in enumerate(sources,1):
        relative=path.relative_to(run).with_suffix('').as_posix()
        name=re.sub(r'[^A-Za-z0-9_-]+','_',relative)+'.png'
        if any(name==n for _,n in included):name=Path(name).stem+'_'+hashlib.sha256(relative.encode()).hexdigest()[:8]+'.png'
        shutil.copy2(path,figures/name);included.append((path,name))
    lookup={p.relative_to(run).as_posix():name for p,name in included};used=set()
    lines=[r'\documentclass[10pt]{article}',r'\usepackage[margin=0.8in]{geometry}',r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern,graphicx,booktabs,longtable,amsmath,float}',r'\usepackage[hidelinks]{hyperref}',r'\setlength{\parindent}{0pt}',r'\setlength{\parskip}{5pt}',r'\title{Arm26 Neural Controller: Training and Tracking Evidence}',r'\author{Hussein Zolfaghari}',r'\date{}',r'\begin{document}',r'\maketitle']
    def figure(rel,explanation=None):
        if rel not in lookup:return
        used.add(rel);lines.extend([r'\begin{figure}[H]\centering',r'\includegraphics[width=\linewidth,height=0.69\textheight,keepaspectratio]{overleaf_figures/'+lookup[rel]+'}',r'\caption{'+tex(explanation or caption(Path(rel)))+'}',r'\label{fig:'+Path(lookup[rel]).stem+'}',r'\end{figure}'])
    lines += [r'\section*{1. Purpose and protocol}',
        'A neural network replaces the cable-force command of the compensated PID teacher. The 21--64--64--1 tanh network is trained to imitate recorded teacher forces. During OpenSim testing, learned weights remain fixed and no PID force fallback is used.',
        r'Whole episodes retain their original training, validation, or test labels. Training updates weights; validation chooses the checkpoint; the held-out test set evaluates the selected model. Only accepted, compatible PID episodes are included. The original acceptance gate is motion RMSE $\leq2^\circ$ and full-episode maximum error $\leq5^\circ$; this gate does not imply that every accepted episode has the same accuracy.',
        r'Training minimizes normalized force error, $\mathcal{L}=\frac{1}{N}\sum_i[(\hat F_i-F_i^{\mathrm{PID}})/650]^2$. Scaling uses fixed physical units. Actual actuator commands are clipped to $[0,650]$ N.',
        r'\begin{center}\begin{tabular}{lrr}\toprule Split & Accepted episodes & Time samples\\\midrule']
    for split in ('train','validation','test'):
        rr=[r for r in manifest if r['split']==split];lines.append(tex(split)+f" & {len(rr)} & {sum(r['samples'] for r in rr)}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        'Counts are actual accepted episodes, so percentages can differ from the planned 70/15/15 allocation. Exact episode IDs and hashes are provided in the accompanying split manifest.']
    if not manifest:lines+=['The split manifest is unavailable; split counts and separation cannot be established from this export.']
    lines += [r'\section*{2. Learning and force prediction}']
    if info.get('evaluation_only_repeat'):
        lines.append('This is a new frozen-network test evaluation. Training curves and force-imitation metrics are reused from the original fitted model; no weights were updated during this repeat.')
    if history:
        best=min(history,key=lambda r:r['validation_force_rmse_N']);lines += [f"Training logged {len(history)} epochs. The lowest logged validation force RMSE was {number(best['validation_force_rmse_N'])} N at epoch {int(best['epoch'])}. This is a force-imitation measure, not an angle-tracking measure."]
    lines += [r'\begin{center}\begin{tabular}{lrrrr}\toprule Split & RMSE (N) & MAE (N) & Max. error (N) & $R^2$\\\midrule']
    for key in ('training','validation','test'):
        m=metrics.get(key)
        if m:lines.append(tex(key)+' & '+' & '.join(number(m.get(k)) for k in ('rmse_N','mae_N','max_error_N','r2'))+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    figure('learning_curve.png');figure('test_parity.png');figure('test_residuals.png')
    lines += [r'\section*{3. Closed-loop OpenSim tracking}']
    if rows:
        completed=sum(r.get('nn_status')=='completed' for r in rows);accepted=sum(bool(r.get('nn_accepted')) for r in rows)
        valid=[r for r in rows if r.get('nn_motion_rmse_deg') is not None and r.get('teacher_motion_rmse_deg') is not None]
        lines.append(f"{completed}/{len(rows)} attempted episodes completed; {accepted}/{len(rows)} met the configured acceptance gate. Failed attempts remain in the results. Motion RMSE is evaluated during motion, whereas full maximum error also includes the initial transient and holds.")
        if valid:
            pid=np.mean([r['teacher_motion_rmse_deg'] for r in valid]);nn=np.mean([r['nn_motion_rmse_deg'] for r in valid]);maximum=max(r['nn_max_error_deg'] for r in valid if r.get('nn_max_error_deg') is not None)
            lines.append(f"The arithmetic mean of episode motion RMSEs was {number(pid)} degrees for the recorded teacher and {number(nn)} degrees for the NN; these are not sample-pooled RMSEs. The largest recorded NN full-episode error was {number(maximum)} degrees. These figures summarize the available {len(valid)} paired episodes.")
        figure('tracking_evidence.png')
        # One example per task / shoulder / trajectory, chosen by sorted ID before inspecting errors.
        seen=set()
        for r in rows:
            group=(r.get('task'),r.get('shoulder_deg'),r.get('trajectory'))
            if group in seen:continue
            seen.add(group);rel='tracking/'+r['episode_id']+'/comparison.png'
            figure(rel,'Example '+r['episode_id']+'. '+caption(Path(rel)))
        lines += [r'\subsection*{Every attempted test episode}',r'\begin{longtable}{p{0.44\linewidth}rrrr}',r'\toprule Episode & PID RMSE & NN RMSE & NN max. & Pass\\\midrule\endhead']
        for r in rows:lines.append(tex(r['episode_id'])+' & '+' & '.join(number(r.get(k)) for k in ('teacher_motion_rmse_deg','nn_motion_rmse_deg','nn_max_error_deg'))+' & '+('Yes' if r.get('nn_accepted') else 'No')+r'\\')
        lines += [r'\bottomrule\end{longtable}', 'All errors in the table are degrees. The full saved trajectories and command histories remain in the original results folder.']
    else:lines += ['Closed-loop tracking results are not available in this run folder. Force prediction results alone do not establish successful control.']
    if audit:
        lines += [r'\section*{4. Duration and sampling check}',f"The export inspected signals from {len(audit)} attempted episodes. Positive, increasing recorded time was found in {sum(r['monotonic_time'] for r in audit)} episodes. The accompanying sampling audit lists every sample count, time span, median interval, configured interval, and recorded computation time.",
            'Runtime alone is not evidence of correctness. The sampling audit reports recorded values; it does not independently prove that every configured final time was reached.']
        figure('duration_audit.png')
    lines += [r'\section*{Scope of the conclusion}',
        'The evidence supports performance on held-out episodes within the collected trajectory families and feasible configurations. It does not establish arbitrary new-trajectory generalization, real-world performance, or superiority over PID. A single training seed is not a repeated-seed robustness study. States outside training feature ranges are reported in the original comparison file.',
        'Run status: '+tex(status.get('status','not recorded'))+'. Report creation alone is not a success criterion.',
        r'\clearpage\appendix\section*{Appendix: complete saved figure set}',
        'Every remaining saved PNG is included below, with a short explanation. This appendix retains all episode diagnostics; the main text presents learning curves, test force accuracy, and tracking evidence.']
    for path,name in included:
        rel=path.relative_to(run).as_posix()
        if rel not in used:figure(rel,rel+': '+caption(path))
    lines += [r'\end{document}'];(export/'main.tex').write_text('\n'.join(lines),encoding='utf-8')
    # Provide an importable body for an existing Overleaf document, as well as a standalone main.tex.
    start=lines.index(r'\maketitle')+1
    (export/'results.tex').write_text('\n'.join(lines[start:-1]),encoding='utf-8')
    (export/'main.tex').write_text('\n'.join(lines[:start]+[r'\input{results}',r'\end{document}']),encoding='utf-8')
    for name in ('split_manifest.json','force_metrics.json','history.csv','run_info.json','pipeline_status.json'):
        if (run/name).exists():shutil.copy2(run/name,export/name)
    if comparison:(export/'tracking_metrics.json').write_text(json.dumps(comparison,indent=2))
    (export/'figure_manifest.json').write_text(json.dumps([dict(source=str(p.relative_to(run)),export='overleaf_figures/'+n,caption=caption(p)) for p,n in included],indent=2))
    (export/'README.txt').write_text('Upload OVERLEAF_REPORT.zip as an Overleaf project and compile main.tex with pdfLaTeX. Edit results.tex for the report body, captions, and figure calls. The overleaf_figures folder contains descriptive stable filenames. To use your existing report, copy results.tex and overleaf_figures into that project, add graphicx, float, booktabs, longtable, amsmath to its preamble, then input results.tex where required. Alternatively copy individual figure calls. The appendix includes remaining saved figures. Re-exporting regenerates the local LaTeX files; preserve your own edits separately. No training or simulation is started by this exporter.\n')
    target=run/'OVERLEAF_REPORT.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(export.rglob('*')):
            if f.is_file() and 'figures' not in f.relative_to(export).parts and f.suffix not in ('.aux','.log','.out','.pdf'):z.write(f,f.relative_to(export))
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-folder',type=Path);a=p.parse_args()
    from saved_run import resolve_saved_run
    a.run_folder=resolve_saved_run(a.run_folder)
    print('SAVED RESULTS:',a.run_folder,flush=True)
    print('OVERLEAF REPORT:',build_report(a.run_folder),flush=True)


if __name__=='__main__':main()
