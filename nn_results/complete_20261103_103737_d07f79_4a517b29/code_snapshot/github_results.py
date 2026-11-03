"""Upload saved results through an isolated standard Git clone, add, commit and push."""
import argparse,hashlib,json,os,shutil,subprocess,tempfile,time,uuid,urllib.request,urllib.parse
from pathlib import Path,PurePosixPath

REPOSITORY='hzolfaghari2022/Continual-Learning-with-human-in-the-loop'
DEFAULT_URL='https://github.com/'+REPOSITORY+'.git'
BRANCH='main'
CHUNK_BYTES=32*1024*1024
ROOT=Path(__file__).resolve().parent


def git(directory,*args,input=None):
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='auto')
    # Keep the isolated upload independent of VS Code/parent repository and date overrides.
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_COMMON_DIR','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES','GIT_NAMESPACE','GIT_AUTHOR_NAME','GIT_AUTHOR_EMAIL','GIT_AUTHOR_DATE','GIT_COMMITTER_NAME','GIT_COMMITTER_EMAIL','GIT_COMMITTER_DATE'):
        env.pop(key,None)
    proc=subprocess.run(['git','-C',str(directory),*args],input=input,text=True,encoding='utf-8',errors='replace',capture_output=True,env=env,timeout=300)
    if proc.returncode:raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or 'Git command failed.')
    return proc.stdout.strip()


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def api_get(path):
    url='https://api.github.com/repos/'+REPOSITORY+'/'+path
    separator='&' if '?' in url else '?'
    url+=separator+'arm26_check='+str(time.time_ns())
    request=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'Arm26-Results-Verifier','Cache-Control':'no-cache'})
    with urllib.request.urlopen(request,timeout=45) as response:return json.load(response)


def verify_api(commit,branch,manifest_path,manifest_sha):
    """Independently verify the committed manifest through the public GitHub API."""
    details=api_get('commits/'+urllib.parse.quote(commit,safe=''))
    if details.get('sha')!=commit:raise RuntimeError('GitHub API commit SHA does not match the uploaded commit.')
    path=urllib.parse.quote(manifest_path,safe='/')
    remote=api_get('contents/'+path+'?ref='+urllib.parse.quote(commit,safe=''))
    if remote.get('type')!='file' or remote.get('sha')!=manifest_sha:raise RuntimeError('GitHub API could not confirm the exact uploaded result manifest.')
    reference=api_get('git/ref/heads/'+urllib.parse.quote(branch,safe=''))
    return dict(verified_manifest_url=remote.get('html_url'),api_branch_head=reference.get('object',{}).get('sha'),commit_author_login=(details.get('author') or {}).get('login'),commit_author_email=details.get('commit',{}).get('author',{}).get('email'),commit_author_date=details.get('commit',{}).get('author',{}).get('date'))


def upload(run_folder,repo_url=DEFAULT_URL,branch=BRANCH,remote_prefix='nn_results'):
    run=Path(run_folder).resolve()
    if not run.is_dir() or not (run/'run_info.json').exists():raise ValueError('Choose a result folder containing run_info.json.')
    if not shutil.which('git'):raise RuntimeError('Git for Windows is missing from PATH.')
    if not remote_prefix or remote_prefix.startswith('/') or '\\' in remote_prefix or any(p in ('.','..') for p in PurePosixPath(remote_prefix).parts):raise ValueError('Invalid upload prefix.')
    identity=run/'github_run_id.txt'
    if not identity.exists():identity.write_text(run.name+'_'+uuid.uuid4().hex[:8])
    run_id=identity.read_text().strip()
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):raise ValueError('Invalid saved run ID.')
    dest=remote_prefix.rstrip('/')+'/'+run_id;status_path=run/'github_upload_status.json'
    def record(**kw):status_path.write_text(json.dumps(kw,indent=2),encoding='utf-8')
    record(status='uploading',repository=REPOSITORY,branch=branch,destination=dest,uploader_version='standard-worktree-v1')
    cache=Path.home()/'Arm26_Git_Uploads';cache.mkdir(exist_ok=True)
    print('GITHUB [1/5]: Cloning current',branch,'for',REPOSITORY,flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix='up_',dir=cache) as temp:
            work=Path(temp);repo=work/'repo'
            git(work,'-c','core.longpaths=true','clone','--quiet','--depth=1','--branch',branch,'--',repo_url,str(repo))
            git(repo,'config','core.longpaths','true');git(repo,'config','core.autocrlf','false');git(repo,'config','commit.gpgsign','false')
            hooks=work/'empty_hooks';hooks.mkdir();git(repo,'config','core.hooksPath',str(hooks))
            git(repo,'config','user.name','Hussein Zolfaghari');git(repo,'config','user.email','105059650+hzolfaghari2022@users.noreply.github.com')
            target=repo.joinpath(*PurePosixPath(dest).parts)
            if not target.resolve().is_relative_to(repo.resolve()):raise ValueError('Upload destination escapes isolated clone.')
            if target.exists():shutil.rmtree(target)
            target.mkdir(parents=True)
            print('GITHUB [2/5]: Copying saved results and source snapshot',flush=True)
            manifest=[]
            files=[(p,p.relative_to(run).as_posix()) for p in sorted(run.rglob('*')) if p.is_file() and not p.is_symlink() and '.git' not in p.relative_to(run).parts and '__pycache__' not in p.parts and p!=status_path]
            files += [(p,'code_snapshot/'+p.name) for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in ('.py','.bat','.txt','.osim') and not p.name.startswith('LATEST_')]
            for p,relative in files:
                if any(c in relative for c in ('\n','\r','\t')):raise ValueError('Unsupported filename control character.')
                size=p.stat().st_size;entry=dict(path=relative,size_bytes=size,sha256=digest(p),parts=[])
                if size<=CHUNK_BYTES:
                    output=target.joinpath(*PurePosixPath(relative).parts);output.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,output)
                else:
                    with p.open('rb') as f:
                        i=0
                        while True:
                            block=f.read(CHUNK_BYTES)
                            if not block:break
                            rel='.large_files/'+relative+f'.part{i:05d}';output=target.joinpath(*PurePosixPath(rel).parts);output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(block);entry['parts'].append(rel);i+=1
                manifest.append(entry)
            (target/'upload_manifest.json').write_text(json.dumps(dict(format_version=1,files=manifest,note='Files above 32 MiB are losslessly split. Run restore_large_files.py after downloading this complete folder.'),indent=2),encoding='utf-8')
            shutil.copy2(ROOT/'restore_large_files.py',target/'restore_large_files.py')
            (target/'GITHUB_README.md').write_text('# Arm26 neural controller results\n\nSaved model, figures, raw test trajectories, logs, HTML and editable Overleaf reports.\n\nFiles above 32 MiB are under .large_files. Download this full folder and run python restore_large_files.py to reassemble and verify hashes. The source snapshot records this uploader package, which may differ from the original training code for recovered runs. Original checkpoint and history are preserved.\n',encoding='utf-8')
            # Preserve numerical files and archives byte-for-byte on Windows.
            (target/'.gitattributes').write_text('* -text\n',encoding='ascii')
            print('GITHUB [3/5]: Staging',len(manifest),'files using git add',flush=True)
            git(repo,'add','--force','--all','--',dest)
            manifest_path=dest+'/upload_manifest.json'
            git(repo,'ls-files','--error-unmatch','--',manifest_path)
            manifest_sha=git(repo,'rev-parse',':'+manifest_path)
            expected_sha=git(repo,'hash-object','--no-filters','--',str(target/'upload_manifest.json'))
            if manifest_sha!=expected_sha:raise RuntimeError('Staged manifest bytes do not match the copied result manifest.')
            changed=bool(git(repo,'diff','--cached','--name-only','--',dest))
            base=git(repo,'rev-parse','HEAD')
            if changed:
                git(repo,'commit','--quiet','-m','Save Arm26 NN results: '+run_id,'--',dest)
                commit=git(repo,'rev-parse','HEAD')
                if commit==base:raise RuntimeError('Git did not create a new commit despite staged result changes.')
            else:commit=base
            # Never push or report success if the new commit lacks the result folder.
            git(repo,'cat-file','-e',commit+':'+manifest_path)
            if git(repo,'rev-parse',commit+':'+manifest_path)!=expected_sha:raise RuntimeError('Committed manifest is different from staged results.')
            print('GITHUB [4/5]: Pushing verified commit',commit,flush=True)
            for attempt in range(3):
                try:
                    git(repo,'push','origin','HEAD:refs/heads/'+branch);break
                except RuntimeError as error:
                    if attempt==2 or not any(word in str(error).lower() for word in ('non-fast-forward','fetch first','rejected')):raise
                    git(repo,'fetch','--quiet','origin',branch)
                    git(repo,'rebase','FETCH_HEAD');commit=git(repo,'rev-parse','HEAD')
                    git(repo,'cat-file','-e',commit+':'+manifest_path)
                    if git(repo,'rev-parse',commit+':'+manifest_path)!=expected_sha:raise RuntimeError('Manifest changed during rebase.')
            remote=git(repo,'ls-remote','origin','refs/heads/'+branch).split()[0]
            if remote!=commit:
                git(repo,'fetch','--quiet','origin',branch);git(repo,'merge-base','--is-ancestor',commit,'FETCH_HEAD')
            print('GITHUB [5/5]: Confirming result manifest through GitHub API',flush=True)
            proof=verify_api(commit,branch,manifest_path,expected_sha)
            url='https://github.com/'+REPOSITORY+'/tree/'+commit+'/'+dest
            record(status='success',repository=REPOSITORY,branch=branch,destination=dest,commit=commit,url=url,files=len(manifest),new_commit_created=changed,github_api_verified=True,uploader_version='standard-worktree-v1',**proof)
            print('GITHUB VERIFIED SUCCESS:',url,flush=True)
            print('COMMIT:',commit,'AUTHOR:',proof['commit_author_login'],'DATE:',proof['commit_author_date'],flush=True)
            print('New commit created.' if changed else 'No new commit: unchanged results were already present.',flush=True)
            return url
    except Exception as error:
        record(status='failed',repository=REPOSITORY,branch=branch,destination=dest,error=str(error),uploader_version='standard-worktree-v1',retry='python upload_results.py --run-folder "'+str(run)+'"')
        print('GITHUB UPLOAD/VERIFICATION FAILED:',error,'\nLocal results are retained. Retry upload_results.py; no training or simulation rerun is needed.',flush=True)
        raise

def maybe_upload(run,args):
    if getattr(args,'no_github',False):return
    try:upload(run,repo_url=getattr(args,'github_url',DEFAULT_URL))
    except Exception as error:
        print('GITHUB: Results were not uploaded:',error,'; use upload_results.py to retry.',flush=True)
