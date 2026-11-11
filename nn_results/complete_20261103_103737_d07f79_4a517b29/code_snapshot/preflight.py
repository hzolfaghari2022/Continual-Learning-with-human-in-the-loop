"""Check dependencies and all split metadata before expensive work."""
import runtime_setup
import argparse,json,hashlib,importlib,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['training','simulation','report'],required=True);p.add_argument('--dataset',type=Path);p.add_argument('--output',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    if a.stage=='simulation':
        import opensim
        model=Path(__file__).resolve().parent/'arm26_exosuit.osim'
        opensim.Model(str(model));print('OpenSim model and bindings: OK');return
    if a.stage=='report':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig=plt.figure();plt.close(fig);print('Plotting: OK');return
    import torch,numpy as np
    from dataset_io import load_episodes
    suite=a.dataset.resolve()
    if not a.allow_partial:
        plan=suite/'episode_plan.json'
        if not plan.exists():raise ValueError('The completed dataset episode_plan.json is missing.')
        missing=[r['episode_id'] for r in json.loads(plan.read_text()) if not (suite/r['episode_id']/'status.json').exists()]
        if missing:raise ValueError(f'Collection is incomplete: {len(missing)} episode statuses missing.')
    all_ids=set();records=[];quality=[]
    for split in ('train','validation','test'):
        x,y,ids,rr=load_episodes(suite,'pooled',split)
        if all_ids&set(ids):raise ValueError('Episode leakage between splits.')
        all_ids.update(ids);records.extend(rr)
        for r in rr:
            state=json.loads((suite/r['episode_id']/'status.json').read_text());motion=state.get('motion_only') or {};full=state.get('all_samples') or {}
            if state.get('status')!='completed' or motion.get('rmse_deg',float('inf'))>2 or full.get('max_abs_error_deg',float('inf'))>5:raise ValueError('An accepted PID episode does not satisfy the documented quality gate: '+r['episode_id'])
            quality.append(dict(episode_id=r['episode_id'],split=split,teacher_motion_rmse_deg=motion['rmse_deg'],teacher_full_max_error_deg=full['max_abs_error_deg']))
        print(f'DATA: {split}: {len(rr)} accepted episodes, {len(x)} samples',flush=True)
    if torch.cuda.is_available():
        device='cuda:0';z=torch.ones(2,device=device);assert float(z.sum().cpu())==2
    else:device='cpu'
    result=dict(dataset=str(suite),device=device,records=records,teacher_quality=quality,dataset_fingerprint=hashlib.sha256(json.dumps(dict(records=records,settings_and_status=[(r['episode_id'],hashlib.sha256((suite/r['episode_id']/'settings.json').read_bytes()).hexdigest(),hashlib.sha256((suite/r['episode_id']/'status.json').read_bytes()).hexdigest()) for r in records]),sort_keys=True).encode()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2));print('Training preflight OK. Device:',device,flush=True)

if __name__=='__main__':main()
