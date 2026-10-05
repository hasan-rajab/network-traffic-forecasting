"""Package verified derived public data, without raw records or model binaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.serve import validate_artifacts


def main() -> None:
    source = Path('.').resolve()
    summary = validate_artifacts(source)
    destination = source / 'deployment/artifacts'
    if destination.exists():
        shutil.rmtree(destination)
    files = [source / 'data/processed/traffic.parquet',
             source / 'data/processed/analytics.sqlite']
    files += [p for p in (source / 'results').iterdir()
              if p.is_file() and p.suffix in {'.json', '.csv', '.parquet'}]
    if sum(p.stat().st_size for p in files) > 50_000_000:
        raise ValueError('Derived serving bundle exceeds the 50 MB limit')
    checksums = {}
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        checksums[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    validate_artifacts(destination)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    provenance = {'source_commit': commit, 'github_run_id': os.getenv('GITHUB_RUN_ID'),
                  'source_doi': '10.7910/DVN/EGZHFV', 'data_license': 'ODbL-1.0',
                  'scope': 'Derived 30-cell analytics and evaluation outputs; raw records excluded',
                  'validated': summary, 'sha256': checksums}
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (destination / 'DATA_LICENSE.md').write_text(
        '# Derived data attribution\n\n'
        'Derived from Telecom Italia, Telecommunications - SMS, Call, Internet - MI.\n'
        'Canonical source: https://doi.org/10.7910/DVN/EGZHFV\n\n'
        'The derived database is provided under the Open Database License 1.0:\n'
        'https://opendatacommons.org/licenses/odbl/1-0/\n\n'
        'The public Kaggle transport mirror is dkgmgo/telecom-italia-milan.\n'
        'Injected anomaly labels are simulated on held-out real traffic.\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
