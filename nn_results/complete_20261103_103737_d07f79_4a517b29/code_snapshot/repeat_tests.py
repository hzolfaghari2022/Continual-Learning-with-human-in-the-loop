"""Repeat OpenSim tests with a fixed saved NN, without training or validation selection."""
import runtime_setup
import argparse,json,shutil,sys,time,uuid,webbrowser,hashlib
from pathlib import Path
from train_and_report import run
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-folder',type=Path,help='Original trained run. Defaults to the last complete-pipeline run.')
    p.add_argument('--no-live',action='store_true');p.add_argument('--no-github',action='store_true')
    p.add_argument('--github-url',default='https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git')
    p.add_argument('--max-episodes',type=int,help='Optional demonstration subset; default repeats every accepted test episode.')
    a=p.parse_args()
    from saved_run import resolve_saved_run
    try:source=resolve_saved_run(a.run_folder)
    except FileNotFoundError as error:p.error(str(error))
    required=['policy.pt','policy_inference.npz','policy_inference.json','history.csv','force_metrics.json','run_info.json']
    missing=[name for name in required if not (source/name).exists()]
    if missing:p.error('Missing trained run files: '+', '.join(missing))
    meta=json.loads((source/'policy_inference.json').read_text());dataset=Path(meta['dataset'])
    if not dataset.is_dir():p.error('Original PID dataset is not available at '+str(dataset))
    out=Path.home()/'Arm26_NN_Results'/('test_repeat_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    # Reuse the existing learning evidence, never invoke train_nn.py.
    copy_names=required+['train_plot_data.npz','validation_plot_data.npz','test_plot_data.npz','split_manifest.json','preflight.json','TRAINING_COMPLETE.json']
    for name in copy_names:
        if (source/name).exists():shutil.copy2(source/name,out/name)
    for path in source.glob('*.png'):shutil.copy2(path,out/path.name)
    info=json.loads((out/'run_info.json').read_text());info.update(evaluation_only_repeat=True,original_training_run=str(source),checkpoint=str(out/'policy.pt'),checkpoint_sha256=hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest());(out/'run_info.json').write_text(json.dumps(info,indent=2))
    print('FROZEN MODEL:',source/'policy.pt',flush=True);print('NEW TEST RESULTS:',out,flush=True)
    command=[sys.executable,str(ROOT/'evaluate_nn.py'),'--dataset',str(dataset),'--checkpoint',str(out/'policy.pt'),'--output',str(out/'tracking'),'--defer-report']
    if a.no_live:command+=['--no-live']
    if a.max_episodes is not None:command+=['--max-episodes',str(a.max_episodes)]
    try:
        rc=run(command,out/'tracking_console.log')
        comparison=out/'tracking'/'comparison.json'
        if rc not in (0,2) or not comparison.exists():raise RuntimeError('OpenSim test evaluation failed; read tracking_console.log.')
        rows=json.loads(comparison.read_text())['rows'];passed=all(r['nn_accepted'] for r in rows)
        (out/'pipeline_status.json').write_text(json.dumps(dict(status='completed',evaluation_only_repeat=True,tracking_all_accepted=passed,evaluated_episodes=len(rows),original_training_run=str(source)),indent=2))
        if run([sys.executable,str(ROOT/'finalize_report.py'),'--run-folder',str(out)],out/'report_console.log'):raise RuntimeError('Test report export failed; read report_console.log.')
        if not a.no_github:
            if run([sys.executable,str(ROOT/'upload_results.py'),'--run-folder',str(out),'--github-url',a.github_url],out/'github_console.log'):raise RuntimeError('Upload failed; retry upload_results.py for this new folder without repeating tests.')
        print('TEST REPEAT COMPLETE. No weights were updated.',flush=True);print('REPORT:',out/'REPORT.html',flush=True);print('OVERLEAF:',out/'OVERLEAF_REPORT.zip',flush=True)
        if not a.no_live:webbrowser.open((out/'REPORT.html').as_uri())
        return 0 if passed else 2
    except (Exception,KeyboardInterrupt) as error:
        print('TEST REPEAT STOPPED:',error,'\nSaved outputs:',out,flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
