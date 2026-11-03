"""Same trained tanh network, evaluated without importing PyTorch into OpenSim."""
import runtime_setup
import json,math
from pathlib import Path
import numpy as np
from policy_features import FEATURES,encode


def load_policy(path,device='cpu'):
    path=Path(path)
    weights=path if path.name.endswith('_inference.npz') else path.with_name(path.stem+'_inference.npz')
    meta=weights.with_suffix('.json')
    if not weights.exists() or not meta.exists():
        raise FileNotFoundError('Inference export missing for '+str(path)+'. Use recover_and_report.py --run-folder YOUR_PREVIOUS_RUN, or train with the updated code.')
    saved=json.loads(meta.read_text(encoding='utf-8'))
    if saved.get('features')!=FEATURES:raise ValueError('Inference feature schema mismatch.')
    with np.load(weights,allow_pickle=False) as z:
        layers=[(np.array(z[f'W{i}']),np.array(z[f'b{i}'])) for i in range(3)]
    if [w.shape for w,b in layers]!=[(64,21),(64,64),(1,64)]:raise ValueError('Unexpected exported network dimensions.')
    def forward(x):
        x=np.asarray(x,dtype=np.float32)
        for i,(w,b) in enumerate(layers):
            x=x@w.T+b
            if i<2:x=np.tanh(x)
        return x
    return forward,saved

class NeuralCableController:
    def __init__(self,o,model,elbow,cable,cfg):
        self.model,self.elbow,self.cfg=model,elbow,cfg
        self.network,self.saved=load_policy(cfg['neural_checkpoint'])
        if abs(self.saved['force_cap_N']-cfg['force_cap_N'])>1e-9:raise ValueError('Force-cap mismatch.')
        self.integral=0.;self.previous=0.;self.ood_samples=0
        self.scales=np.array(self.saved['scales'],dtype=np.float32)
        self.low=np.array(self.saved['training_min'],dtype=np.float32);self.high=np.array(self.saved['training_max'],dtype=np.float32)

    def step(self,state,qr,vr,ar,dt,arm):
        cfg=self.cfg;self.model.realizeDynamics(state)
        q=self.elbow.getValue(state);v=self.elbow.getSpeedValue(state);memory=self.integral
        mus=self.model.getMuscles()
        x=[q,v,qr,vr,ar,cfg['mass_kg'],math.radians(cfg['shoulder_deg']),self.previous,memory]
        x += [mus.get(i).getActivation(state) for i in range(mus.getSize())]
        x += [mus.get(i).getFiberLength(state)/mus.get(i).getOptimalFiberLength() for i in range(mus.getSize())]
        x=np.asarray(x,dtype=np.float32)
        if not np.isfinite(x).all():raise FloatingPointError('Nonfinite NN input.')
        if np.any((x<self.low-1e-4*self.scales)|(x>self.high+1e-4*self.scales)):self.ood_samples+=1
        raw=float(self.network(encode(x,self.scales) if self.saved.get('encoding')=='tracking_error_v1' else x/self.scales).item())*cfg['force_cap_N']
        if not math.isfinite(raw):raise FloatingPointError('Nonfinite NN force command.')
        force=float(np.clip(raw,0.,cfg['force_cap_N']));error=qr-q
        if not ((raw>cfg['force_cap_N'] and error>0) or (raw<0 and error<0)):
            self.integral=float(np.clip(memory+error*dt,-cfg['integral_limit_rad_s'],cfg['integral_limit_rad_s']))
        self.previous=force
        return dict(force_N=force,requested_force_N=raw,torque_Nm=raw*arm,
            desired_acceleration_rad_s2='not_used',P_accel='not_used',I_accel='not_used',D_accel='not_used',
            error_integral_rad_s=memory,saturated=abs(raw-force)>1e-7)
