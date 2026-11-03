"""Reassemble large GitHub result files, verifying their saved SHA-256 hashes."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent


def safe(relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Manifest path escapes this folder.')
    return path


def main():
    manifest=json.loads((root/'upload_manifest.json').read_text())
    for r in manifest['files']:
        if not r['parts']:continue
        target=safe(r['path']);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.reassembling');h=hashlib.sha256();size=0
        try:
            with tmp.open('wb') as output:
                for rel in r['parts']:
                    with safe(rel).open('rb') as f:
                        for data in iter(lambda:f.read(1024*1024),b''):h.update(data);size+=len(data);output.write(data)
            if h.hexdigest()!=r['sha256'] or size!=r['size_bytes']:raise ValueError('File hash/size mismatch: '+r['path'])
            tmp.replace(target);print('Restored:',r['path'])
        finally:tmp.unlink(missing_ok=True)

if __name__=='__main__':main()
