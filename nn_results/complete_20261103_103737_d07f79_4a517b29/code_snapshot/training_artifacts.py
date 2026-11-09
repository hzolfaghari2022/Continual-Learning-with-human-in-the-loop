"""Write report inputs and portable inference weights, without plotting imports."""
import json
from pathlib import Path
import numpy as np


def export_weights(checkpoint,saved):
    checkpoint=Path(checkpoint);state=saved['state_dict'];arrays={}
    for i,k in enumerate((0,2,4)):
        arrays[f'W{i}']=state[f'net.{k}.weight'].detach().cpu().numpy()
        arrays[f'b{i}']=state[f'net.{k}.bias'].detach().cpu().numpy()
    target=checkpoint.with_name(checkpoint.stem+'_inference.npz')
    np.savez_compressed(target,**arrays)
    metadata={k:v for k,v in saved.items() if k not in ('state_dict','fisher')}
    target.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


def export_plot_data(out,splits):
    out=Path(out)
    for name,(x,y,p,records,ids) in splits.items():
        np.savez_compressed(out/(name+'_plot_data.npz'),y=y,p=p,ids=ids,
                            records_json=np.array(json.dumps(records)))
