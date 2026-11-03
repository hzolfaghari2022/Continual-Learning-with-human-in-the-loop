"""One command: verify data, train, test OpenSim, report, and push to GitHub."""
import runtime_setup
import argparse,hashlib,json,subprocess,sys,time,uuid,webbrowser,shutil
from pathlib import Path
from dataset_io import resolve_dataset
from train_and_report import run
ROOT=Path(__file__).resolve().parent


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--no-live',action='store_true');p.add_argument('--no-github',action='store_true')
    p.add_argument('--github-url',default='https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git')
    p.add_argument('--new-run',action='store_true',help='Start fresh instead of continuing an incomplete pipeline.')
    p.add_argument('--output',type=Path);p.add_argument('--allow-partial',action='store_true',help='Exploratory use only.')
    p.add_argument('--max-episodes',type=int,help='Exploratory test subset; default evaluates every accepted test episode.')
    a=p.parse_args()
    if a.epochs<1:p.error('epochs must be positive')
    suite=resolve_dataset(a.dataset)
    base=Path.home()/'Arm26_NN_Results';base.mkdir(exist_ok=True);pointer=base/'LATEST_COMPLETE_PIPELINE.txt'
    previous=Path(pointer.read_text().strip()) if pointer.exists() else None
    if a.output:out=a.output.resolve()
    elif not a.new_run and previous and (previous/'workflow_state.json').exists() and read(previous/'workflow_state.json').get('status')!='completed':out=previous
    else:out=base/('complete_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=True);pointer.write_text(str(out));state=read(out/'workflow_state.json',dict(status='started',stages={}))
    configuration=dict(dataset=str(suite),epochs=a.epochs,allow_partial=a.allow_partial,max_episodes=a.max_episodes)
    if state.get('configuration') and state['configuration']!=configuration:p.error('Incomplete run has a different dataset/configuration. Keep its original arguments, or use --new-run.')
    state['configuration']=configuration;state['status']='running';write(out/'workflow_state.json',state)
    print('RESULTS:',out,flush=True);print('DATASET:',suite,flush=True)
    stage='preflight'
    try:
        print('[1/5] CHECKING DATA, DEPENDENCIES AND GITHUB',flush=True)
        # Each dependency family is checked in its own clean process.
        command=[sys.executable,str(ROOT/'preflight.py'),'--stage','training','--dataset',str(suite),'--output',str(out/'preflight.json')]
        if a.allow_partial:command+=['--allow-partial']
        if run(command,out/'preflight_training.log'):raise RuntimeError('Training/data preflight failed. Read preflight_training.log.')
        check=read(out/'preflight.json')
        if state.get('dataset_fingerprint') and state['dataset_fingerprint']!=check['dataset_fingerprint']:raise ValueError('Dataset changed since the previous attempt; automatic reuse is blocked. Use the unchanged data or --new-run.')
        state['dataset_fingerprint']=check['dataset_fingerprint']
        for kind in ('simulation','report'):
            if run([sys.executable,str(ROOT/'preflight.py'),'--stage',kind],out/('preflight_'+kind+'.log')):raise RuntimeError(kind+' dependency preflight failed.')
        if not a.no_github:
            if not shutil.which('git'):raise RuntimeError('Git for Windows is missing from PATH. Restart VS Code after installing it.')
            proc=subprocess.run(['git','ls-remote','--exit-code',a.github_url,'refs/heads/main'],capture_output=True,text=True,timeout=120)
            if proc.returncode:raise RuntimeError('GitHub connection/sign-in check failed before training: '+proc.stderr.strip())
        state['stages']['preflight']='completed';write(out/'workflow_state.json',state)
        stage='training';print('[2/5] TRAINING + VALIDATION + FINAL FORCE TEST',flush=True)
        marker=read(out/'TRAINING_COMPLETE.json',{})
        reusable=marker and all((out/f).exists() for f in ('policy.pt','policy_inference.npz','policy_inference.json','history.csv','force_metrics.json','run_info.json','train_plot_data.npz','validation_plot_data.npz','test_plot_data.npz'))
        if reusable and marker.get('checkpoint_sha256')!=hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest():raise ValueError('Saved checkpoint changed; refusing automatic reuse.')
        if reusable:print('Using completed training from this same pipeline; no weight updates repeated.',flush=True)
        else:
            cmd=[sys.executable,str(ROOT/'train_nn.py'),'--dataset',str(suite),'--task','pooled','--epochs',str(a.epochs),'--output',str(out),'--defer-report']
            if a.allow_partial:cmd+=['--allow-partial']
            if run(cmd,out/'training_console.log'):raise RuntimeError('Training failed; read training_console.log.')
            if not (out/'TRAINING_COMPLETE.json').exists():raise RuntimeError('Training completion record missing.')
        state['stages']['training']='completed';write(out/'workflow_state.json',state)
        stage='tracking';print('[3/5] ACTUAL NN-ONLY OPENSIM TEST ROLLOUTS',flush=True)
        cmd=[sys.executable,str(ROOT/'evaluate_nn.py'),'--dataset',str(suite),'--checkpoint',str(out/'policy.pt'),'--output',str(out/'tracking'),'--resume-episodes','--defer-report']
        if a.no_live:cmd+=['--no-live']
        if a.max_episodes is not None:cmd+=['--max-episodes',str(a.max_episodes)]
        rc=run(cmd,out/'tracking_console.log');comparison=read(out/'tracking'/'comparison.json',{})
        if rc not in (0,2) or not comparison.get('rows'):raise RuntimeError('Tracking evaluation failed; read tracking_console.log.')
        all_pass=all(r.get('nn_accepted',False) for r in comparison['rows'])
        write(out/'pipeline_status.json',dict(status='completed',tracking_all_accepted=all_pass,evaluated_episodes=len(comparison['rows']),note='Scientific results are complete. Workflow upload/report status is stored separately in workflow_state.json.'))
        state['stages']['tracking']='completed';state['tracking_all_accepted']=all_pass;write(out/'workflow_state.json',state)
        stage='report';print('[4/5] GENERATING FIGURES, HTML AND OVERLEAF ZIP',flush=True)
        if run([sys.executable,str(ROOT/'finalize_report.py'),'--run-folder',str(out)],out/'report_console.log'):raise RuntimeError('Report export failed. The saved checkpoint and completed tracking episodes are retained.')
        state['stages']['report']='completed';write(out/'workflow_state.json',state)
        stage='github';print('[5/5] UPLOADING ALL RESULTS TO GITHUB' if not a.no_github else '[5/5] GITHUB DISABLED BY --no-github',flush=True)
        if not a.no_github:
            if run([sys.executable,str(ROOT/'upload_results.py'),'--run-folder',str(out),'--github-url',a.github_url],out/'github_console.log'):raise RuntimeError('GitHub upload failed. Rerun the same run_all.py command to continue without retraining.')
            upload=read(out/'github_upload_status.json',{})
            if upload.get('status')!='success':raise RuntimeError('GitHub success receipt missing.')
            state['github_url']=upload['url'];state['stages']['github']='completed'
        else:state['stages']['github']='disabled'
        state['status']='completed';write(out/'workflow_state.json',state)
        print('\nWORKFLOW COMPLETE',flush=True);print('TRACKING:', 'all passed' if all_pass else 'some tests failed; see measured errors in report',flush=True)
        print('REPORT:',out/'REPORT.html',flush=True);print('OVERLEAF:',out/'OVERLEAF_REPORT.zip',flush=True)
        if state.get('github_url'):print('GITHUB:',state['github_url'],flush=True)
        if not a.no_live:webbrowser.open((out/'REPORT.html').as_uri())
        return 0 if all_pass else 2
    except (Exception,KeyboardInterrupt) as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed';state['failed_stage']=stage;state['error']=str(error);write(out/'workflow_state.json',state)
        print('STOPPED AT:',stage,'\nDETAIL:',error,'\nSaved work:',out,'\nRerun the same command to continue. Completed training and matching completed rollouts are reused.',flush=True)
        return 1

if __name__=='__main__':raise SystemExit(main())
