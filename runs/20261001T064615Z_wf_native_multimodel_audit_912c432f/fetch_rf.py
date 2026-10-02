import urllib.request,json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent/'artifacts/rf';r.mkdir(parents=True,exist_ok=True)
base='https://api.github.com/repos/robust-fingerprinting/RF'
def get(u):return urllib.request.urlopen(u,timeout=30).read()
meta=json.loads(get(base));rev=json.loads(get(base+'/commits/'+meta['default_branch']))['sha'];tree=json.loads(get(base+'/git/trees/'+rev+'?recursive=1'));(r/'tree.json').write_text(json.dumps(tree,indent=2));print('COMMIT',rev,flush=True)
manifest=[]
for entry in tree['tree']:
 p=entry['path']
 if entry['type']=='blob' and (p.endswith('.py') or p.lower().endswith(('readme.md','requirements.txt','license'))):
  if entry.get('size',0)>250000:continue
  u='https://raw.githubusercontent.com/robust-fingerprinting/RF/'+rev+'/'+p;b=get(u);target=r/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b);manifest.append({'path':p,'url':u,'sha256':hashlib.sha256(b).hexdigest()});print(p,flush=True)
(r/'provenance.json').write_text(json.dumps({'repo':meta['html_url'],'commit':rev,'files':manifest},indent=2)+'\n')
