"""Package the completed submission without data, checkpoints or generated caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if (HERE.parent/'scripts').exists() else HERE.parent.parent
OUT=ROOT/'submission_2A202602810'
SKIP_PARTS={'__pycache__','.ipynb_checkpoints'}
SKIP_FILES={'run.log','predictions_baseline.csv'}
SKIP_SUFFIXES={'.pyc','.pt','.pth','.ckpt','.npz'}

if __name__=='__main__':
    target=ROOT/(OUT.name+'.zip')
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and not SKIP_PARTS.intersection(p.parts) and p.name not in SKIP_FILES and p.suffix not in SKIP_SUFFIXES:
                z.write(p,p.relative_to(ROOT))
    print(target)
