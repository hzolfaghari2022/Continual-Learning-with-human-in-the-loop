"""Approved preview references, with analytical time derivatives and speed scaling."""
import math
import numpy as np
FAMILIES=['smooth_lift','smooth_oscillation','multiple_targets']

def base(family,t):
    t=float(np.clip(t,0,8))
    if family=='smooth_oscillation':
        w=math.pi/2
        return 45-25*math.cos(w*t),25*w*math.sin(w*t),25*w*w*math.cos(w*t)
    knots=[0,4,8] if family=='smooth_lift' else [0,1.6,3.2,4.8,6.4,8]
    angles=[20,80,20] if family=='smooth_lift' else [20,45,75,35,65,20]
    i=min(len(knots)-2,max(0,int(np.searchsorted(knots,t,side='right'))-1))
    T=knots[i+1]-knots[i];u=(t-knots[i])/T;d=angles[i+1]-angles[i]
    return (angles[i]+d*(10*u**3-15*u**4+6*u**5),
            d*(30*u*u-60*u**3+30*u**4)/T,
            d*(60*u-180*u*u+120*u**3)/T**2)

def peak_speed_deg_s(family):
    if family=='smooth_oscillation':return 25*math.pi/2
    if family=='smooth_lift':return 1.875*60/4
    return 1.875*max(25,30,40,30,45)/1.6

def configure(cfg,radius):
    # Keep the approved shape; stretch time to satisfy each task's peak hand speed.
    scale=math.radians(peak_speed_deg_s(cfg['trajectory']))*radius/cfg['peak_hand_speed_m_s']
    cfg['trajectory_time_scale']=scale
    cfg['motion_duration_s']=8*scale
    cfg['duration_s']=cfg['start_hold_s']+cfg['motion_duration_s']+cfg['end_hold_s']
    return cfg

def reference(t,cfg):
    elapsed=t-cfg['start_hold_s'];scale=cfg['trajectory_time_scale']
    if elapsed<0 or elapsed>cfg['motion_duration_s']:
        return math.radians(20),0.,0.,False
    q,v,a=base(cfg['trajectory'],elapsed/scale)
    return math.radians(q),math.radians(v)/scale,math.radians(a)/scale**2,True
