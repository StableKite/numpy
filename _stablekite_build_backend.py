"""PEP 517 delivery of verified native wheels, with an explicit source build.

CPython 3.15t x64 Windows/Linux uses the exact qualified release artifact.
Other source builds continue through NumPy's original meson-python backend.
No NumPy module is imported while resolving or preparing package metadata.
"""
from email.parser import BytesParser
from hashlib import sha256
from pathlib import Path, PurePosixPath
import ast
import importlib
import json
import os
import platform
import shutil
import struct
import sys
import sysconfig
import tempfile
import tomllib
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
SOURCE_OPTION = 'stablekite-source-build'


def _source_requested(config_settings):
    value = (config_settings or {}).get(SOURCE_OPTION, 'false')
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f'{SOURCE_OPTION} must occur once')
        value = value[0]
    value = str(value).lower()
    if value not in {'true', '1', 'false', '0'}:
        raise ValueError(f'{SOURCE_OPTION} must be true or false')
    return value in {'true', '1'}


def _meson_settings(config_settings):
    return {k: v for k, v in (config_settings or {}).items() if k != SOURCE_OPTION}


def _platform_key():
    if (sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 15)
            or not sysconfig.get_config_var('Py_GIL_DISABLED') or struct.calcsize('P') != 8
            or platform.machine().lower() not in {'amd64', 'x86_64'}):
        return None
    if sys.platform == 'win32':
        return 'win_amd64'
    if sys.platform == 'linux' and platform.libc_ver()[0] == 'glibc':
        return 'linux_x86_64_glibc'
    return None


def _carrier(config_settings):
    return not _source_requested(config_settings) and _platform_key() is not None


def _project():
    with (ROOT / 'pyproject.toml').open('rb') as stream:
        return tomllib.load(stream)


def _artifact():
    manifest = json.loads((ROOT / 'stablekite-wheels.json').read_bytes())
    version = _project()['project']['version']
    if manifest['version'] != version or manifest['schema_version'] != 1:
        raise RuntimeError('NumPy release manifest and source project disagree')
    key = _platform_key()
    artifact = manifest['artifacts'].get(key)
    if artifact is None:
        raise RuntimeError(f'No qualified NumPy wheel for {key}; use -C{SOURCE_OPTION}=true to build the source')
    if key == 'linux_x86_64_glibc':
        actual = tuple(map(int, platform.libc_ver()[1].split('.')))
        required = tuple(map(int, artifact['minimum_glibc'].split('.')))
        if actual < required:
            raise RuntimeError(f'This NumPy wheel requires glibc {artifact["minimum_glibc"]}; use -C{SOURCE_OPTION}=true to build the source')
    if (PurePosixPath(artifact['filename']).name != artifact['filename']
            or not artifact['filename'].endswith('-' + artifact['tag'] + '.whl')
            or not artifact['url'].startswith('https://github.com/StableKite/numpy/releases/download/')
            or len(artifact['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in artifact['sha256'])):
        raise RuntimeError('Invalid NumPy release artifact declaration')
    return manifest, artifact


def _validate_wheel(path, manifest, artifact):
    if path.stat().st_size != artifact['size'] or sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
        raise RuntimeError('NumPy release wheel SHA-256 or size mismatch')
    with ZipFile(path) as wheel:
        metadata_names = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata_names) != 1:
            raise RuntimeError('NumPy wheel must contain one METADATA file')
        metadata_name, = metadata_names
        metadata = BytesParser().parsebytes(wheel.read(metadata_name))
        if metadata['Name'] != 'numpy' or metadata['Version'] != manifest['version']:
            raise RuntimeError('NumPy wheel metadata differs from the declared source version')
        dist_info = metadata_name.rsplit('/', 1)[0]
        wheel_metadata = BytesParser().parsebytes(wheel.read(dist_info + '/WHEEL'))
        if artifact['tag'] not in wheel_metadata.get_all('Tag', []):
            raise RuntimeError('NumPy wheel ABI/platform tag mismatch')
        version_tree = ast.parse(wheel.read('numpy/version.py'))
        constants = {n.targets[0].id: n.value.value for n in version_tree.body
                     if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
        if (constants.get('version') != manifest['version']
                or constants.get('git_revision') != manifest['native_source_commit']):
            raise RuntimeError('NumPy runtime version/source commit differs from its release declaration')
        native_hashes = artifact['native_hashes']
        if not any('/_multiarray_umath.' in n for n in native_hashes):
            raise RuntimeError('NumPy core binary is absent from the release declaration')
        for name, expected in native_hashes.items():
            if sha256(wheel.read(name)).hexdigest() != expected:
                raise RuntimeError(f'NumPy native binary hash mismatch: {name}')
    return dist_info


def _wheel():
    manifest, artifact = _artifact()
    cache = Path(tempfile.gettempdir()) / 'stablekite-numpy-wheel-cache'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (artifact['sha256'] + '.whl')
    if path.exists():
        _validate_wheel(path, manifest, artifact)
        return path, manifest, artifact
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, suffix='.download', delete=False) as destination:
            temporary = Path(destination.name)
            request = urllib.request.Request(artifact['url'], headers={'User-Agent': 'StableKite-NumPy-PEP517'})
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, destination)
        _validate_wheel(temporary, manifest, artifact)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path, manifest, artifact


def _source_requirements():
    requirements = list(_project()['tool']['stablekite']['source-build-requires'])
    if sys.version_info[:2] >= (3, 15):
        requirements = [r for r in requirements if not r.startswith('Cython')]
        requirements.append('Cython==3.3.1')
    return requirements


def get_requires_for_build_wheel(config_settings=None):
    return [] if _carrier(config_settings) else _source_requirements()


def _extract_metadata(path, metadata_directory):
    with ZipFile(path) as wheel:
        metadata_names = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata_names) != 1:
            raise RuntimeError('NumPy wheel must contain one METADATA file')
        metadata_name, = metadata_names
        metadata = BytesParser().parsebytes(wheel.read(metadata_name))
        if metadata['Name'] != 'numpy' or metadata['Version'] != _project()['project']['version']:
            raise RuntimeError('Prepared source metadata differs from the NumPy project')
        dist_info = metadata_name.rsplit('/', 1)[0]
        for name in wheel.namelist():
            if not name.startswith(dist_info + '/') or name.endswith('/'):
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Invalid path in NumPy wheel metadata')
            destination = Path(metadata_directory).joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(wheel.read(name))
    return dist_info


def _source_cached_wheel(metadata_directory):
    if metadata_directory is None:
        return None
    metadata_directory = Path(metadata_directory)
    for parent in (metadata_directory, metadata_directory.parent):
        folder = parent / '_stablekite_source_wheel'
        if not (folder / 'RECEIPT.json').exists():
            continue
        receipt = json.loads((folder / 'RECEIPT.json').read_bytes())
        if (receipt['pyproject_sha256'] != sha256((ROOT / 'pyproject.toml').read_bytes()).hexdigest()
                or Path(receipt['filename']).name != receipt['filename']):
            raise RuntimeError('NumPy source-build metadata cache mismatch')
        path = folder / receipt['filename']
        if sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
            raise RuntimeError('NumPy prepared source wheel hash mismatch')
        return path
    return None


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    if not _carrier(config_settings):
        meson = importlib.import_module('mesonpy')
        hook = getattr(meson, 'prepare_metadata_for_build_wheel', None)
        if hook is not None:
            return hook(metadata_directory, _meson_settings(config_settings))
        # meson-python has no metadata-only hook. Preserve PEP 517's normal
        # build-once fallback and reuse those exact bytes in build_wheel.
        folder = Path(metadata_directory) / '_stablekite_source_wheel'
        folder.mkdir(parents=True, exist_ok=True)
        filename = meson.build_wheel(str(folder), _meson_settings(config_settings))
        if Path(filename).name != filename:
            raise RuntimeError('Invalid meson-python wheel filename')
        path = folder / filename
        receipt = {'filename': filename, 'sha256': sha256(path.read_bytes()).hexdigest(),
                   'pyproject_sha256': sha256((ROOT / 'pyproject.toml').read_bytes()).hexdigest()}
        (folder / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return _extract_metadata(path, metadata_directory)
    path, manifest, artifact = _wheel()
    _validate_wheel(path, manifest, artifact)
    return _extract_metadata(path, metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    if not _carrier(config_settings):
        cached = _source_cached_wheel(metadata_directory)
        if cached is not None:
            Path(wheel_directory).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, Path(wheel_directory) / cached.name)
            return cached.name
        return importlib.import_module('mesonpy').build_wheel(
            wheel_directory, _meson_settings(config_settings), metadata_directory)
    path, manifest, artifact = _wheel()
    Path(wheel_directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, Path(wheel_directory) / artifact['filename'])
    print(f'Using verified native NumPy {manifest["version"]} wheel; SHA256 {artifact["sha256"]}', flush=True)
    return artifact['filename']


def get_requires_for_build_sdist(config_settings=None):
    return _source_requirements()


def build_sdist(sdist_directory, config_settings=None):
    return importlib.import_module('mesonpy').build_sdist(sdist_directory, _meson_settings(config_settings))


def get_requires_for_build_editable(config_settings=None):
    return _source_requirements()


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return importlib.import_module('mesonpy').build_editable(
        wheel_directory, _meson_settings(config_settings), metadata_directory)
