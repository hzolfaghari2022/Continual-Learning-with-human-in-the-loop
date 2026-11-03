"""One entry point. Reuse a trained model by default; train only with --train-new."""
import runtime_setup
import argparse,hashlib,json,os,re,subprocess,sys,webbrowser,zipfile
from pathlib import Path
from saved_run import resolve_saved_run
from train_and_report import run
ROOT=Path(__file__).resolve().parent


def verify_overleaf(folder):
    project=folder/'overleaf_report';archive=folder/'OVERLEAF_REPORT.zip'
    for path in (project/'main.tex',project/'results.tex',project/'overleaf_figures',archive):
        if not path.exists():raise FileNotFoundError('Required Overleaf output missing: '+str(path))
    body=(project/'results.tex').read_text(encoding='utf-8')
    refs=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',body)
    if not refs:raise ValueError('No figures referenced in the LaTeX report.')
    for relative in refs:
        path=(project/relative).resolve()
        if not path.is_relative_to(project.resolve()) or not path.is_file():raise ValueError('Broken LaTeX figure reference: '+relative)
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        if not {'main.tex','results.tex'}.issubset(names) or any(p not in names for p in refs):raise ValueError('Overleaf ZIP does not contain every referenced figure/source.')
    return project,len(refs)


def show(folder):
    project=folder/'overleaf_report'
    print('\nHTML REPORT:',folder/'REPORT.html',flush=True)
    print('OVERLEAF MAIN:',project/'main.tex',flush=True)
    print('EDITABLE TEXT:',project/'results.tex',flush=True)
    print('FIGURE FOLDER:',project/'overleaf_figures',flush=True)
    print('UPLOAD THIS ZIP TO OVERLEAF:',folder/'OVERLEAF_REPORT.zip',flush=True)
    webbrowser.open((folder/'REPORT.html').as_uri())
    if os.name=='nt':
        try:os.startfile(str(project))
        except OSError as error:print('Open this folder manually:',project,'(',error,')',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-folder',type=Path,help='Existing trained result; automatically located when omitted.')
    p.add_argument('--train-new',action='store_true',help='Explicitly train a NEW model from scratch.')
    p.add_argument('--repeat-tests',action='store_true',help='Rerun OpenSim tests with frozen weights; no training.')
    p.add_argument('--no-live',action='store_true');p.add_argument('--no-github',action='store_true')
    p.add_argument('--dataset',type=Path);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--github-url',default='https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git')
    a=p.parse_args();out=None
    if a.train_new and (a.run_folder or a.repeat_tests):p.error('--train-new is separate from reusing an existing run.')
    try:
        if a.train_new:
            print('[1/4] EXPLICIT NEW TRAINING: training, validation and all OpenSim test episodes',flush=True)
            command=[sys.executable,str(ROOT/'train_new_pipeline.py'),'--new-run','--no-github','--epochs',str(a.epochs)]
            if a.dataset:command+=['--dataset',str(a.dataset.resolve())]
            if a.no_live:command+=['--no-live']
            rc=subprocess.run(command,cwd=ROOT).returncode
            if rc not in (0,2):raise RuntimeError('New training/testing stopped. Saved outputs remain in Arm26_NN_Results; do not repeat training to solve a report/upload issue.')
            out=resolve_saved_run()
        else:
            out=resolve_saved_run(a.run_folder)
            print('[1/4] REUSING YOUR TRAINED MODEL:',out/'policy.pt',flush=True)
            print('No fitting or weight updates will run.',flush=True)
        original_hash=hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest()
        required=['policy_inference.json','policy_inference.npz','train_plot_data.npz','validation_plot_data.npz','test_plot_data.npz']
        if any(not (out/f).exists() for f in required):
            print('Regenerating missing exports from saved weights (no training).',flush=True)
            if run([sys.executable,str(ROOT/'export_saved_run.py'),'--run-folder',str(out)],out/'export_console.log'):raise RuntimeError('Saved-model export failed. Read export_console.log; no retraining was started.')
        metadata=json.loads((out/'policy_inference.json').read_text());dataset=Path(metadata['dataset'])
        comparison=out/'tracking'/'comparison.json'
        if a.repeat_tests or not comparison.exists():
            # The evaluator resumes completed identical episodes unless a fresh repeat was explicitly requested.
            target=out/'tracking'
            if a.repeat_tests:
                import time,uuid
                target=out/('tracking_repeat_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
            command=[sys.executable,str(ROOT/'evaluate_nn.py'),'--dataset',str(dataset),'--checkpoint',str(out/'policy.pt'),'--output',str(target),'--resume-episodes','--defer-report']
            if a.no_live:command+=['--no-live']
            if run(command,out/'test_console.log') not in (0,2) or not (target/'comparison.json').exists():raise RuntimeError('OpenSim testing failed; saved training is retained.')
            if a.repeat_tests:
                # Report the new repeat without replacing the original tracking data.
                import shutil,time,uuid
                fresh=out.parent/('test_report_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);fresh.mkdir()
                for name in ['policy.pt','policy_inference.json','policy_inference.npz','history.csv','force_metrics.json','run_info.json','split_manifest.json','preflight.json','train_plot_data.npz','validation_plot_data.npz','test_plot_data.npz']:
                    if (out/name).exists():shutil.copy2(out/name,fresh/name)
                shutil.copytree(target,fresh/'tracking');info=json.loads((fresh/'run_info.json').read_text());info.update(evaluation_only_repeat=True,original_training_run=str(out));(fresh/'run_info.json').write_text(json.dumps(info,indent=2));out=fresh
        rows=json.loads((out/'tracking'/'comparison.json').read_text())['rows']
        passed=all(r.get('nn_accepted',False) for r in rows)
        (out/'pipeline_status.json').write_text(json.dumps(dict(status='completed',training_reused=not a.train_new,tracking_all_accepted=passed,evaluated_episodes=len(rows)),indent=2))
        print('[2/4] GENERATING ALL FIGURES AND EDITABLE OVERLEAF REPORT',flush=True)
        if run([sys.executable,str(ROOT/'finalize_report.py'),'--run-folder',str(out)],out/'report_console.log'):raise RuntimeError('Report generation failed; inspect report_console.log. Your saved model is retained.')
        project,count=verify_overleaf(out)
        if hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest()!=original_hash:raise ValueError('Unexpected checkpoint change during reporting/testing.')
        print('[3/4] VERIFIED:',count,'LaTeX figure references; main.tex, results.tex, overleaf_figures and ZIP are present.',flush=True)
        if not a.no_live:show(out)
        else:
            print('OVERLEAF FOLDER:',project,flush=True);print('OVERLEAF ZIP:',out/'OVERLEAF_REPORT.zip',flush=True)
        print('[4/4] UPLOADING REPORTS, FIGURES AND SAVED RESULTS' if not a.no_github else '[4/4] GITHUB DISABLED BY YOUR OPTION',flush=True)
        if not a.no_github:
            if run([sys.executable,str(ROOT/'upload_results.py'),'--run-folder',str(out),'--github-url',a.github_url],out/'github_console.log'):raise RuntimeError('GitHub upload failed. Reports already exist at the paths printed above; no retraining is needed.')
        (ROOT/'LATEST_RESULTS.txt').write_text(str(out))
        print('\nDONE: SAVED MODEL, TEST RESULTS, FIGURES AND OVERLEAF REPORT ARE READY.',flush=True)
        if not passed:print('Some tracking tests did not pass; the report retains their measured results.',flush=True)
        return 0 if passed else 2
    except (Exception,KeyboardInterrupt) as error:
        print('STOPPED:',error,flush=True)
        if out:print('YOUR RESULTS ARE SAFE:',out,flush=True)
        print('No training is started by default. Use --train-new only when you intentionally want a new model.',flush=True)
        return 1

if __name__=='__main__':raise SystemExit(main())
