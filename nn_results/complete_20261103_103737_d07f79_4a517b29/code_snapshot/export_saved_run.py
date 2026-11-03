"""Recover artifacts from an already selected checkpoint, without updating weights."""
import runtime_setup
import argparse,csv,json
from pathlib import Path
import torch
from train_nn import load_episodes,force_metrics,predict
from neural_policy import load_policy
from training_artifacts import export_weights,export_plot_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-folder',type=Path,required=True);p.add_argument('--dataset',type=Path);a=p.parse_args()
    out=a.run_folder.resolve();checkpoint=out/'policy.pt'
    if not checkpoint.exists():raise FileNotFoundError('No saved policy.pt in '+str(out)+'. This run cannot be recovered without training.')
    torch.set_num_threads(2);model,saved=load_policy(checkpoint)
    suite=(a.dataset or Path(saved['dataset'])).resolve();task=saved['stage'];splits={};metrics={};all_ids=set()
    for name,key in [('train','training'),('validation','validation'),('test','test')]:
        try:x,y,ids,records=load_episodes(suite,task,name)
        except ValueError as error:
            if name!='test' or not str(error).startswith('No accepted '):raise
            metrics['test_unavailable']=str(error);continue
        if name!='test':
            original=saved[key+'_records']
            expected={(r['episode_id'],r['sha256']) for r in original}
            actual={(r['episode_id'],r['sha256']) for r in records}
            if expected!=actual:raise ValueError('Saved checkpoint '+name+' data IDs/hashes differ. Use the original unchanged dataset.')
        if all_ids&set(ids):raise ValueError('Episode overlap between splits.')
        all_ids.update(ids);pred=predict(model,x,'cpu');metrics[key]=force_metrics(y,pred)
        splits[name]=(x,y,pred,records,ids)
        if name=='test':metrics['test_records']=records
    metrics['note']='Recovered existing selected weights; no training or checkpoint selection. Test is evaluated after selection. Tracking requires OpenSim.'
    export_weights(checkpoint,saved);export_plot_data(out,splits)
    (out/'force_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    # The previous code saved history before importing plotting libraries.
    with (out/'history.csv').open() as f:history=list(csv.DictReader(f))
    (out/'run_info.json').write_text(json.dumps(dict(dataset=str(suite),task=task,device='CPU artifact recovery; original training may use CUDA',seed=saved['seed'],epochs_completed=saved.get('epochs_completed',len(history)),partial_collection=saved.get('partial_collection'),checkpoint=str(checkpoint),recovered_without_retraining=True),indent=2))
    print('RECOVERED:',checkpoint,flush=True)


if __name__=='__main__':main()
