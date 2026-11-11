"""Set process-local MKL policy before scientific imports; never suppress duplicate-runtime checks."""
import os
# Keep NumPy's MKL from loading its threaded OpenMP runtime beside PyTorch's.
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
