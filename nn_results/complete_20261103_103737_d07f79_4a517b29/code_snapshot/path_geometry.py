"""Geometry calculations for logging; no controller or inverse dynamics."""
import numpy as np
def path_moment_arms(o,model,state,elbow,cable,muscles,delta=1e-5):
    """Signed -d(path length)/dq with shoulder held at its actual fixed position.
    Avoids OpenSim MomentArmSolver's constraint/cache failure at nonzero locked
    shoulder angles. Central differences apply to original cable/muscle paths.
    """
    q=elbow.getValue(state);lengths=[]
    for sign in [-1.,1.]:
        probe=o.State(state);elbow.setValue(probe,q+sign*delta,False)
        model.realizePosition(probe)
        lengths.append([cable.getLength(probe)]+[muscles.get(i).getLength(probe) for i in range(muscles.getSize())])
    return -(np.array(lengths[1])-np.array(lengths[0]))/(2*delta)
