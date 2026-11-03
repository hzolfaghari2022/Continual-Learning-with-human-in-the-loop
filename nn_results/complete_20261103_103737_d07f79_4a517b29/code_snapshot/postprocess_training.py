"""Separate plotting process: never imports PyTorch."""
import runtime_setup
import argparse,csv,json
from pathlib import Path
import numpy as np
from reporting import training_figures,write_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-folder',type=Path,required=True);a=p.parse_args();out=a.run_folder.resolve()
    with (out/'history.csv').open() as f:history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    splits={}
    for name in ('train','validation','test'):
        path=out/(name+'_plot_data.npz')
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:
            # Figures only require sample count, targets, predictions, episode IDs.
            splits[name]=(np.empty((len(z['y']),0)),z['y'],z['p'],json.loads(str(z['records_json'])),z['ids'])
    training_figures(out,history,splits,json.loads((out/'force_metrics.json').read_text()))
    print('REPORT:',write_report(out),flush=True)


if __name__=='__main__':main()
