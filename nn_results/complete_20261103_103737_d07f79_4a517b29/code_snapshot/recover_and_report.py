"""Continue a saved run after a plotting/runtime failure. Does not train."""
import runtime_setup
import argparse,json,sys,webbrowser,shutil,time
from pathlib import Path
from train_and_report import run
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-folder',type=Path,required=True);p.add_argument('--dataset',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--no-live',action='store_true');p.add_argument('--max-episodes',type=int)
    p.add_argument('--no-github',action='store_true',help='Keep this run local; automatic GitHub upload is on by default.')
    p.add_argument('--github-url',default='https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git')
    a=p.parse_args();source=a.run_folder.resolve()
    if not (source/'policy.pt').exists():p.error('policy.pt was not found in that run folder. Check the path before trying training again.')
    out=(a.output or Path.home()/'Arm26_NN_Results'/('recovered_'+time.strftime('%Y%m%d_%H%M%S'))).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if out!=source:
        for name in ['policy.pt','history.csv']:
            if not (source/name).exists():p.error('Recovery needs '+str(source/name))
            shutil.copy2(source/name,out/name)
    print('RECOVERY SOURCE:',source,flush=True);print('RESULTS:',out,flush=True)
    status={};stage='checkpoint export'
    try:
        cmd=[sys.executable,str(ROOT/'export_saved_run.py'),'--run-folder',str(out)]
        if a.dataset:cmd+=['--dataset',str(a.dataset.resolve())]
        if run(cmd,out/'recovery_console.log'):raise RuntimeError('Checkpoint export failed; read recovery_console.log.')
        stage='plotting'
        if run([sys.executable,str(ROOT/'postprocess_training.py'),'--run-folder',str(out)],out/'plotting_console.log'):raise RuntimeError('Plotting failed; read plotting_console.log.')
        saved=json.loads((out/'policy_inference.json').read_text());suite=(a.dataset or Path(saved['dataset'])).resolve()
        stage='OpenSim tracking'
        cmd=[sys.executable,str(ROOT/'evaluate_nn.py'),'--dataset',str(suite),'--checkpoint',str(out/'policy.pt'),'--output',str(out/'tracking')]
        if a.no_live:cmd+=['--no-live']
        if a.max_episodes is not None:cmd+=['--max-episodes',str(a.max_episodes)]
        rc=run(cmd,out/'tracking_console.log')
        if rc not in (0,2) or not (out/'tracking'/'comparison.json').exists():raise RuntimeError('Tracking failed; read tracking_console.log.')
        status=dict(status='completed',recovered_without_retraining=True,tracking_all_accepted=rc==0)
        (ROOT/'LATEST_POLICY.txt').write_text(str(out/'policy.pt'));(ROOT/'LATEST_NN_DATASET.txt').write_text(str(suite))
    except KeyboardInterrupt:status=dict(status='interrupted',stage=stage)
    except Exception as error:status=dict(status='failed',stage=stage,error=str(error));print('ERROR:',error,flush=True)
    (out/'pipeline_status.json').write_text(json.dumps(status,indent=2))
    try:
        from reporting import write_report
        target=write_report(out);print('OPEN REPORT:',target,flush=True)
        if not a.no_live:webbrowser.open(target.as_uri())
    except Exception as error:print('Report unavailable:',error,flush=True)
    from github_results import maybe_upload
    maybe_upload(out,a)
    return 0 if status.get('status')=='completed' and status.get('tracking_all_accepted') else 2


if __name__=='__main__':raise SystemExit(main())
