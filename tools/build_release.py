"""Archive the committed public source only; never include local output/cache."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neo_sigma import __version__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise SystemExit('Commit the intended source changes before building a release.')
    target = args.destination.resolve()
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f'sigma-rescale-neo-{__version__}-preview.zip'
    if archive.exists():
        raise SystemExit(f'Archive already exists: {archive}')
    subprocess.run(['git', 'archive', '--format=zip', '--prefix=sd-forge-neo-sigma-rescale/',
                    '-o', str(archive), 'HEAD'], cwd=root, check=True)
    with zipfile.ZipFile(archive) as bundle:
        files = {name: hashlib.sha256(bundle.read(name)).hexdigest()
                 for name in bundle.namelist() if not name.endswith('/')}
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {'archive': archive.name, 'sha256': digest,
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'files': files}
    archive.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(json.dumps({'archive': str(archive), 'files': len(files), 'sha256': digest}))


if __name__ == '__main__':
    main()
