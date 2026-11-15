"""Retain available package notices from the actual installer build environment."""
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys


def collect(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name = distribution.metadata['Name']
        package = destination / name
        retained = []
        for relative in distribution.files or ():
            base = Path(str(relative)).name.lower()
            if not base.startswith(('license', 'licence', 'copying', 'notice', 'copyright')):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            # Preserve nested license directories without accepting '..' entries.
            parts = [part for part in Path(str(relative)).parts if part not in ('.', '..')]
            target = package.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            retained.append(str(target.relative_to(destination)).replace('\\', '/'))
        inventory.append({
            'name': name, 'version': distribution.version,
            'license_expression': distribution.metadata.get('License-Expression'),
            'license_metadata': distribution.metadata.get('License'),
            'project_urls': distribution.metadata.get_all('Project-URL') or [],
            'retained_notices': retained,
            'review': 'Retained available files; native/transitive notices still need review.'
            if retained else 'No installed license files found; inspect upstream before public release.',
        })
    for candidate in [Path(sys.base_prefix) / 'LICENSE.txt', Path(sys.base_prefix) / 'LICENSE']:
        if candidate.is_file():
            shutil.copyfile(candidate, destination / 'PYTHON-LICENSE.txt')
            break
    (destination / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
    return destination


if __name__ == '__main__':
    collect(sys.argv[1])
