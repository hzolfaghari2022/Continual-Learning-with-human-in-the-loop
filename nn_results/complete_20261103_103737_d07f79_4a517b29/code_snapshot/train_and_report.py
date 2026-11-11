"""Train on saved PID data, then evaluate the frozen NN in OpenSim."""
import runtime_setup
import argparse
import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def run(command,log):
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        try:
            for line in proc.stdout:
                print(line,end='',flush=True);f.write(line);f.flush()
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate();proc.wait();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--no-live',action='store_true')
    p.add_argument('--allow-partial',action='store_true',help='Exploratory training on an unfinished collection only.')
    p.add_argument('--max-episodes',type=int,help='Limit test rollouts; omit for all accepted test episodes.')
    p.add_argument('--output',type=Path)
    p.add_argument('--no-github',action='store_true',help='Keep this run local; automatic GitHub upload is on by default.')
    p.add_argument('--github-url',default='https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git')
    a=p.parse_args()
    out=(a.output or Path.home()/'Arm26_NN_Results'/('report_'+time.strftime('%Y%m%d_%H%M%S'))).resolve();out.mkdir(parents=True,exist_ok=True)
    print('RESULTS:',out,flush=True)
    stage='preflight';status={}
    try:
        from reporting import write_report
        from dataset_io import resolve_dataset
        suite=resolve_dataset(a.dataset)
        print('SAVED PID DATA:',suite,flush=True)
        stage='training'
        command=[sys.executable,str(ROOT/'train_nn.py'),'--dataset',str(suite),'--task','pooled','--epochs',str(a.epochs),'--output',str(out)]
        if a.allow_partial:command+=['--allow-partial']
        rc=run(command,out/'training_console.log')
        if rc:raise RuntimeError(f'Training exited with code {rc}. Read training_console.log for the full error; tracking has not started.')
        stage='OpenSim tracking'
        command=[sys.executable,str(ROOT/'evaluate_nn.py'),'--dataset',str(suite),'--checkpoint',str(out/'policy.pt'),'--output',str(out/'tracking')]
        if a.no_live:command+=['--no-live']
        if a.max_episodes is not None:command+=['--max-episodes',str(a.max_episodes)]
        rc=run(command,out/'tracking_console.log')
        if rc not in (0,2) or not (out/'tracking'/'comparison.json').exists():
            raise RuntimeError(f'OpenSim evaluation exited with code {rc}. Read tracking_console.log and per-episode console.log files.')
        status=dict(status='completed',tracking_all_accepted=rc==0,note='Inspect measured results; completion does not imply every tracking test passed.')
    except KeyboardInterrupt:
        status=dict(status='interrupted',stage=stage);print('Interrupted. Completed outputs are retained.',flush=True)
    except Exception as error:
        status=dict(status='failed',stage=stage,error=str(error));print('ERROR:',str(error),flush=True)
    (out/'pipeline_status.json').write_text(json.dumps(status,indent=2))
    try:
        from reporting import write_report
        report=write_report(out);print('OPEN REPORT:',report,flush=True)
        if not a.no_live:webbrowser.open(report.as_uri())
    except Exception as error:print('Report could not be generated:',error,'; inspect logs in',out,flush=True)
    from github_results import maybe_upload
    maybe_upload(out,a)
    return 0 if status.get('status')=='completed' and status.get('tracking_all_accepted') else 2


if __name__=='__main__':raise SystemExit(main())
