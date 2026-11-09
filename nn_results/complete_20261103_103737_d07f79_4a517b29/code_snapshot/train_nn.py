"""Train direct force imitation: A first, then B plain/EWC, or an explicit pooled baseline."""
import runtime_setup
import argparse,csv,hashlib,json,random,time,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from neural_policy import FEATURES,SCALES,ForceMLP,load_policy,encode

ROOT=Path(__file__).resolve().parent

def resolve_dataset(value=None):
    if value:return Path(value).expanduser().resolve()
    candidates=sorted((Path.home()/'Arm26_Data').glob('*'),reverse=True)
    for suite in candidates:
        for p in suite.glob('*/training_arrays.npz'):
            with np.load(p,allow_pickle=False) as z:
                if list(z['feature_names'])==FEATURES:return suite
    raise FileNotFoundError('No corrected dataset found. Supply --dataset FULL_BATCH_FOLDER, not nn_data.')

def load_episodes(suite,task,split):
    xs=[];ys=[];ids=[];records=[]
    for p in sorted(Path(suite).glob('*/settings.json')):
        folder=p.parent;e=json.loads(p.read_text(encoding='utf-8-sig'))
        if e.get('task') not in (['A','B'] if task=='pooled' else [task]) or e.get('split')!=split:continue
        status=folder/'status.json';arrays=folder/'training_arrays.npz'
        if not status.exists() or not arrays.exists():continue
        r=json.loads(status.read_text())
        if not r.get('accepted_for_imitation',False):continue
        if e.get('controller')!='inverse_dynamics_plus_fixed_PID_acceleration':raise ValueError('Use corrected teacher data only: '+str(folder))
        with np.load(arrays,allow_pickle=False) as z:
            if list(z['feature_names'])!=FEATURES:raise ValueError('Old/mixed feature schema: '+str(arrays))
            x=np.asarray(z['X_raw'],np.float32);y=np.asarray(z['y_force_N'],np.float32).reshape(-1,1)
            if x.shape!=(len(y),21) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid arrays: '+str(arrays))
            if np.min(y)<0 or np.max(y)>650+1e-4:raise ValueError('Force label outside actuator bounds.')
            xs.append(x);ys.append(y/650.);ids.extend([e['episode_id']]*len(y))
        records.append(dict(episode_id=e['episode_id'],task=e['task'],split=split,samples=len(y),sha256=hashlib.sha256(arrays.read_bytes()).hexdigest()))
    if not xs:raise ValueError(f'No accepted {task}/{split} episodes. Finish collection or supply another batch. Splits are not silently reassigned.')
    return np.concatenate(xs),np.concatenate(ys),np.array(ids),records

def force_metrics(y,p):
    delta=(p-y)*650
    denom=float(np.sum((y-y.mean())**2))
    return dict(rmse_N=float(np.sqrt(np.mean(delta**2))),mae_N=float(np.mean(abs(delta))),
        max_error_N=float(abs(delta).max()),r2=None if denom<1e-12 else float(1-np.sum((p-y)**2)/denom))

def predict(model,x,device):
    model.eval();out=[]
    with torch.inference_mode():
        for a in np.array_split(x,max(1,int(np.ceil(len(x)/8192)))):
            out.append(model(torch.from_numpy(encode(a)).to(device)).cpu().numpy())
    return np.concatenate(out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--task',choices=['A','B','pooled'],default='pooled')
    p.add_argument('--init',type=Path);p.add_argument('--ewc-lambda',type=float,default=0.)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--patience',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='auto')
    p.add_argument('--output',type=Path);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--defer-report',action='store_true',help='Pipeline generates plots in a later stage.')
    a=p.parse_args()
    if a.ewc_lambda<0:p.error('EWC lambda must be nonnegative.')
    if a.task=='B' and a.init is None:p.error('Task B requires --init model_A.pt for sequential learning.')
    if a.ewc_lambda and (a.task!='B' or not a.init):p.error('EWC is applied only while learning B from A.')
    if a.epochs<=0 or a.batch_size<=0 or a.patience<=0:p.error('Use positive epochs/batch size/patience.')
    if a.task=='pooled' and a.init:p.error('The pooled baseline starts independently; omit --init.')
    suite=resolve_dataset(a.dataset)
    print('DATASET:',suite,flush=True)
    if not a.allow_partial:
        plan=suite/'episode_plan.json'
        if not plan.exists():raise ValueError('Batch plan missing. Use the full LOCAL completed batch, or explicitly --allow-partial for an exploratory run.')
        expected=json.loads(plan.read_text());missing=[e['episode_id'] for e in expected if not (suite/e['episode_id']/'status.json').exists()]
        if missing:raise ValueError(f'Collection unfinished: {len(missing)} episode records missing. Finish collection before final training; --allow-partial is exploratory only.')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    print('Training device:',device,flush=True)
    x,y,ids,train_records=load_episodes(suite,a.task,'train');vx,vy,vids,val_records=load_episodes(suite,a.task,'validation')
    if set(ids)&set(vids):raise ValueError('Episode overlap between train and validation.')
    if a.init:
        model,initial=load_policy(a.init,device)
        if initial['stage']!='A':raise ValueError('Sequential B must branch from Model A.')
        if initial.get('encoding')!='tracking_error_v1':raise ValueError('Sequential stages must use the same error-feature encoding.')
    else:model=ForceMLP().to(device);initial=None
    anchor={k:v.detach().clone() for k,v in model.named_parameters()}
    fisher=None
    if a.ewc_lambda:
        fisher={k:torch.as_tensor(v,device=device) for k,v in initial['fisher'].items()}
    loader=DataLoader(TensorDataset(torch.from_numpy(encode(x)),torch.from_numpy(y)),batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed))
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    out=(a.output or ROOT/'nn_runs'/f'{a.task}_{"ewc" if a.ewc_lambda else "plain"}_{a.seed}_{time.strftime("%Y%m%d_%H%M%S")}').resolve();out.mkdir(parents=True,exist_ok=True)
    history=[];best=float('inf');stale=0;best_state=None
    for epoch in range(1,a.epochs+1):
        model.train();total=0.;seen=0
        for xb,yb in loader:
            xb=xb.to(device);yb=yb.to(device);optimizer.zero_grad(set_to_none=True)
            pred=model(xb);loss=((pred-yb)**2).mean()
            if fisher is not None:loss=loss+a.ewc_lambda/2*sum((fisher[k]*(v-anchor[k])**2).sum() for k,v in model.named_parameters())
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            total+=float(loss.detach())*len(xb);seen+=len(xb)
        vp=predict(model,vx,device);vl=float(np.mean((vp-vy)**2));history.append(dict(epoch=epoch,train_objective=total/seen,training_force_rmse_N=float(np.sqrt(np.mean((predict(model,x,device)-y)**2))*650),validation_force_rmse_N=float(np.sqrt(vl)*650)))
        print(f'Epoch {epoch}: validation force RMSE={history[-1]["validation_force_rmse_N"]:.4f} N',flush=True)
        if vl<best:best=vl;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=a.patience and a.task!='B':break
    model.load_state_dict(best_state);model.eval()
    # Model-Fisher for unit-variance normalized-force Gaussian regression, TRAIN A ONLY.
    fisher_saved={}
    if a.task=='A':
        fisher_saved={k:torch.zeros_like(v,device='cpu') for k,v in model.named_parameters()}
        selected=np.random.default_rng(a.seed).choice(len(x),size=min(512,len(x)),replace=False)
        for i in selected:
            model.zero_grad(set_to_none=True);model(torch.from_numpy(encode(x[i:i+1])).to(device)).sum().backward()
            for k,v in model.named_parameters():fisher_saved[k]+=v.grad.detach().cpu()**2/len(selected)
    saved=dict(format_version=1,encoding='tracking_error_v1',epochs_requested=a.epochs,epochs_completed=len(history),batch_size=a.batch_size,learning_rate=a.lr,state_dict=best_state,width=64,features=FEATURES,scales=SCALES.tolist(),force_cap_N=650.,
        training_min=x.min(axis=0).tolist(),training_max=x.max(axis=0).tolist(),stage=a.task,seed=a.seed,ewc_lambda=a.ewc_lambda,
        initialization=str(a.init) if a.init else None,fisher=fisher_saved,training_records=train_records,validation_records=val_records,
        partial_collection=a.allow_partial,teacher='inverse_dynamics_plus_fixed_PID_acceleration',dataset=str(suite))
    checkpoint=out/'policy.pt';torch.save(saved,checkpoint)
    from training_artifacts import export_weights,export_plot_data
    export_weights(checkpoint,saved)
    with (out/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    metrics={'training':force_metrics(y,predict(model,x,device)),'validation':force_metrics(vy,predict(model,vx,device)),
        'note':'Force imitation only. Tracking requires closed-loop OpenSim evaluation. Test episodes were not loaded during fitting/checkpoint selection.'}
    splits={'train':(x,y,predict(model,x,device),train_records,ids),'validation':(vx,vy,predict(model,vx,device),val_records,vids)}
    # Final force test is evaluated only after the checkpoint is selected.
    try:
        tx,ty,tids,test_records=load_episodes(suite,a.task,'test')
        if set(tids)&(set(ids)|set(vids)):raise ValueError('Test episode overlap.')
        metrics['test']=force_metrics(ty,predict(model,tx,device))
        metrics['test_records']=test_records
        splits['test']=(tx,ty,predict(model,tx,device),test_records,tids)
    except ValueError as error:
        if not str(error).startswith('No accepted '):raise
        metrics['test_unavailable']=str(error)
    (out/'force_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    (out/'run_info.json').write_text(json.dumps(dict(dataset=str(suite),task=a.task,device=device,seed=a.seed,epochs_completed=len(history),partial_collection=a.allow_partial,checkpoint=str(checkpoint)),indent=2))
    export_plot_data(out,splits)
    (out/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),epochs_completed=len(history)),indent=2))
    if not a.defer_report:
        subprocess.run([sys.executable,str(ROOT/'postprocess_training.py'),'--run-folder',str(out)],check=True)
    (ROOT/'LATEST_POLICY.txt').write_text(str(checkpoint));(ROOT/'LATEST_NN_DATASET.txt').write_text(str(suite))
    print('CHECKPOINT:',checkpoint,flush=True);print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
