import hashlib,json
from pathlib import Path
import numpy as np
from policy_features import FEATURES

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

