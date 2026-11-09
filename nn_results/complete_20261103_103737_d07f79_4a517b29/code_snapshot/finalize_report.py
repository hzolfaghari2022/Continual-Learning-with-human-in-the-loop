"""Final plots, HTML, and Overleaf ZIP; no PyTorch or OpenSim imports."""
import runtime_setup
import argparse,json,zipfile
from pathlib import Path
from reporting import write_report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-folder',type=Path,required=True);a=p.parse_args();out=a.run_folder
    target=write_report(out)
    # write_report logs export errors; the complete pipeline treats them as failures.
    archive=out/'OVERLEAF_REPORT.zip'
    if not archive.exists():raise RuntimeError('Overleaf ZIP was not generated. Read overleaf_export_error.txt.')
    if (out/'overleaf_export_error.txt').exists():raise RuntimeError((out/'overleaf_export_error.txt').read_text())
    with zipfile.ZipFile(archive) as z:
        if 'main.tex' not in z.namelist() or not any(n.startswith('overleaf_figures/') for n in z.namelist()):raise ValueError('Overleaf ZIP is incomplete.')
    manifest=json.loads((out/'split_manifest.json').read_text())
    if not all(any(r['split']==s for r in manifest) for s in ('train','validation','test')):raise ValueError('Missing split in final report.')
    print('FINAL REPORT:',target,flush=True);print('OVERLEAF ZIP:',archive,flush=True)
