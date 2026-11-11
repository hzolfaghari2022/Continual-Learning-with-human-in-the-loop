"""Upload an already completed result folder; no training or simulation."""
import argparse
from pathlib import Path
from github_results import upload,DEFAULT_URL

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-folder',type=Path);p.add_argument('--github-url',default=DEFAULT_URL)
    a=p.parse_args()
    try:
        from saved_run import resolve_saved_run
        a.run_folder=resolve_saved_run(a.run_folder)
        print('SAVED RESULTS TO UPLOAD:',a.run_folder,flush=True)
        upload(a.run_folder,repo_url=a.github_url)
    except Exception as error:
        print('GITHUB: Upload not completed:',error,flush=True)
        raise SystemExit(1)
