#!/usr/bin/env bash
# A code-only snapshot in an independent Git repository, including untracked sources.
set -eu -o pipefail
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
snapshot=${1:?Usage: prepare_three_dataset_code.sh NEW_DIRECTORY}
mkdir -p "$snapshot"
snapshot=$(cd "$snapshot" && pwd -P)
[[ -z "$(ls -A "$snapshot")" ]] || { echo "Use an empty snapshot directory" >&2; exit 1; }
cd "$project"
python - "$snapshot" <<'PY'
from pathlib import Path
import re,shutil,sys
dest=Path(sys.argv[1]); code={'.py','.sh','.sbatch','.ipynb'}
names=[Path(s) for s in ('.gitignore','.gitattributes','README.md','environment.yml','pyproject.toml','method.png')]
names+=list(Path('.').glob('requirements*.txt'))
for folder in ('agmem','scripts','configs','tests','doc','.wheels','reports','runs/analysis'):
    for p in Path(folder).rglob('*'):
        if not p.is_file() or p.is_symlink() or set(p.parts)&{'__pycache__','.pytest_cache','.cache','.git'} or p.suffix in {'.pyc','.tmp'}:
            continue
        if folder in ('reports','runs/analysis') and p.suffix not in code:
            continue
        if folder=='configs' and p.name.startswith('local.') and p.name!='local.example.yaml':
            continue
        if p.name.startswith('.env'):
            continue
        names.append(p)
pattern=re.compile(rb'(?:sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{24,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)')
for p in sorted(set(names)):
    if p.suffix in code|{'.yaml','.yml','.toml','.txt','.md','.json'} and pattern.search(p.read_bytes()):
        raise ValueError(f'Check possible credential in {p}')
    (dest/p.parent).mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest/p)
print('Code files:',len(set(names)))
PY
git -C "$snapshot" init -q -b migration-three-datasets-20261122
git -C "$snapshot" add -f -- .
git -C "$snapshot" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" commit -qm "Snapshot AG-mem code and transfer checklist for three datasets"
git -C "$snapshot" remote add origin https://github.com/BiorelaxA/AG-mem.git
git -C "$snapshot" rev-parse HEAD
