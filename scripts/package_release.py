"""Create deterministic download archives from this public repository only."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
DEST=DOCS/'downloads'
def package(name,files):
    DEST.mkdir(parents=True,exist_ok=True)
    files=sorted({p for p in files if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
    with zipfile.ZipFile(DEST/name,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,10,25,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    b=(DEST/name).read_bytes()
    return {'name':name,'files':len(files),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
    code=list((ROOT/'benchmark_code').rglob('*'))+list((ROOT/'scripts').rglob('*'))+list((ROOT/'licenses').rglob('*'))
    code += [ROOT/n for n in ('build_site.py','README.md','LICENSE','NOTICE','PUBLICATION_BOUNDARY.md')]
    data=[p for p in (DOCS/'data').glob('*') if p.name not in ('download_manifest.json','publication_audit.json')]
    notices=[DOCS/'materials/RIGHTS.md',DOCS/'materials/source_attribution.json',ROOT/'LICENSE',ROOT/'NOTICE']
    results=[package('benchmark-code.zip',code),package('complete-results.zip',data+notices),package('materials.zip',list((DOCS/'materials').rglob('*'))+notices)]
    (DOCS/'data/download_manifest.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
