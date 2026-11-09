"""Direct bounded cable-force policy. No inverse-dynamics/PID command at inference."""
import runtime_setup
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn

from policy_features import FEATURES,SCALES,encode

class ForceMLP(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(21,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,x):return self.net(x)

def load_policy(path,device='cpu'):
    saved=torch.load(Path(path),map_location=device,weights_only=True)
    if saved.get('features')!=FEATURES:raise ValueError('Checkpoint feature names/ordering do not match the corrected collector.')
    model=ForceMLP(saved['width']).to(device);model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved

class NeuralCableController:
    def __init__(self,o,model,elbow,cable,cfg):
        self.model,self.elbow,self.cfg=model,elbow,cfg
        torch.set_num_threads(1)
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
        with torch.inference_mode():raw=float(self.network(torch.from_numpy(encode(x,self.scales) if self.saved.get('encoding')=='tracking_error_v1' else x/self.scales).unsqueeze(0)).item())*cfg['force_cap_N']
        if not math.isfinite(raw):raise FloatingPointError('Nonfinite NN force command.')
        force=float(np.clip(raw,0.,cfg['force_cap_N']));error=qr-q
        if not ((raw>cfg['force_cap_N'] and error>0) or (raw<0 and error<0)):
            self.integral=float(np.clip(memory+error*dt,-cfg['integral_limit_rad_s'],cfg['integral_limit_rad_s']))
        self.previous=force
        return dict(force_N=force,requested_force_N=raw,torque_Nm=raw*arm,
            desired_acceleration_rad_s2='not_used',P_accel='not_used',I_accel='not_used',D_accel='not_used',
            error_integral_rad_s=memory,saturated=abs(raw-force)>1e-7)
