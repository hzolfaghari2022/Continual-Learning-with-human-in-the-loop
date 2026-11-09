import numpy as np

FEATURES=['angle_rad','velocity_rad_s','reference_rad','reference_velocity_rad_s',
'reference_acceleration_rad_s2','mass_kg','shoulder_angle_rad','previous_tension_N','error_integral_rad_s']
MUSCLES=['TRIlong','TRIlat','TRImed','BIClong','BICshort','BRA']
FEATURES += [m+'_activation' for m in MUSCLES]+[m+'_normalized_fiber_length' for m in MUSCLES]
# Fixed unit scales, chosen before training; no validation/test statistics are fitted.
SCALES=np.array([1.,2.,1.,2.,10.,20.,1.,650.,.1]+[1.]*12,dtype=np.float32)

def encode(x,scales=SCALES):
    """Equivalent tracking coordinates expose small errors without changing raw logs."""
    x=np.asarray(x,dtype=np.float32)
    z=x/np.asarray(scales,dtype=np.float32)
    z=np.array(z,copy=True)
    z[...,2]=(x[...,2]-x[...,0])/.02
    z[...,3]=(x[...,3]-x[...,1])/.2
    return z

