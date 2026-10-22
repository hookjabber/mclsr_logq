"""Bounded, hash-only provenance captured before confirmatory training.

The source inventory covers working-tree code (including untracked files), not
just HEAD. It deliberately excludes payload directories, symlinks and common
secret filenames; it is an inventory, not a restorable source archive. Config
secret keys are redacted and the environment uses an explicit allowlist.
Reservation markers survive failures: use a new experiment/seed or working
directory rather than silently reusing an interrupted checkpoint namespace.
The JSON sidecar uses a .provenance suffix so legacy report-only *.json globs
never ingest metadata as an additional experiment result.
Imported irec modules must originate from the inventoried src/irec tree; the
guard reports mismatches without changing sys.path, reloading or replacing imports.
"""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch

DATASET_INPUTS = (
    'train_mclsr.txt',
    'train_sasrec.txt',
    'valid_history.txt',
    'valid_target.txt',
    'test_history.txt',
    'test_target.txt',
)
PREPROCESSING_MANIFESTS = ('preprocessing.json', 'preprocessing_manifest.json', 'manifest.json')
SOURCE_DIRECTORIES = ('src', 'scripts', 'tests', 'configs')
SOURCE_SUFFIXES = {'.py', '.pyi', '.sh', '.json', '.toml', '.yaml', '.yml', '.cfg', '.ini'}
EXCLUDED_DIRECTORIES = {
    'data',
    'datasets',
    'results',
    'checkpoints',
    'tensorboard_logs',
    'logs',
    '__pycache__',
    'node_modules',
    'venv',
}
LOCKFILES = ('uv.lock', 'poetry.lock', 'requirements.txt', 'requirements-dev.txt', 'conda-lock.yml')
MAX_SOURCE_FILES = 10000
MAX_SOURCE_BYTES = 128 * 1024 * 1024
SECRET_NAME = re.compile(r'secret|password|passwd|credential|token|api[_-]?key|private[_-]?key', re.IGNORECASE)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'input changed while capturing provenance: {path}')
    return {'path': str(path), 'sha256': digest, 'size_bytes': after.st_size}


def _redact(value):
    if isinstance(value, dict):
        return {key: '<redacted>' if SECRET_NAME.search(key) else _redact(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value


def validate_imported_sources(source_root, imported_modules=None):
    """Validate actual loaded origins; an explicit module map supports synthetic fixtures."""
    source_root = Path(source_root).resolve()
    package_root = source_root / 'src' / 'irec'
    modules = dict(sys.modules if imported_modules is None else imported_modules)
    if modules.get('irec') is None:
        raise RuntimeError('Cannot verify imported source: irec must be imported before capturing provenance')

    def checked_path(module_name, attribute, value):
        if not isinstance(value, (str, os.PathLike)) or not value:
            raise RuntimeError(f'Cannot verify imported source for {module_name}: missing {attribute}')
        path = Path(value).resolve()
        if not path.is_relative_to(package_root):
            raise RuntimeError(
                f'Imported-source mismatch: {module_name} {attribute} resolves to {path}; '
                f'expected a source under {package_root}. Use a matching environment or set '
                f'PYTHONPATH={source_root / "src"} before starting Python. Imports were not modified.'
            )
        return path

    records = {}
    for name, module in sorted(modules.items()):
        if name != 'irec' and not name.startswith('irec.'):
            continue
        source_file = checked_path(name, '__file__', getattr(module, '__file__', None))
        spec_origin = checked_path(name, '__spec__.origin', getattr(getattr(module, '__spec__', None), 'origin', None))
        if source_file != spec_origin or (name == 'irec' and source_file != package_root / '__init__.py'):
            raise RuntimeError(
                f'Imported-source mismatch: {name} has inconsistent origins '
                f'(__file__={source_file}, __spec__.origin={spec_origin}); expected package {package_root}'
            )
        if not source_file.is_file():
            raise RuntimeError(f'Cannot verify imported source for {name}: source file is unavailable: {source_file}')
        package_paths = [checked_path(name, '__path__', path) for path in getattr(module, '__path__', ())]
        records[name] = {
            'file': str(source_file),
            'spec_origin': str(spec_origin),
            'package_paths': [str(path) for path in package_paths],
        }
    return records


def _git_metadata(source_root):
    def git(*args):
        return subprocess.check_output(
            ['git', '-C', str(source_root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()

    result = {'git_root': None, 'git_commit': None, 'git_dirty': None}
    try:
        result['git_root'] = git('rev-parse', '--show-toplevel')
        result['git_dirty'] = bool(git('status', '--porcelain', '--untracked-files=normal'))
        result['git_commit'] = git('rev-parse', 'HEAD')
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _source_inventory(source_root):
    def allowed_name(name):
        return not name.startswith('.') and name not in EXCLUDED_DIRECTORIES and not SECRET_NAME.search(name)

    candidates = [
        path
        for path in source_root.iterdir()
        if path.is_file() and path.suffix in {'.py', '.pyi', '.sh', '.toml', '.cfg', '.ini'}
    ]
    for directory in SOURCE_DIRECTORIES:
        root = source_root / directory
        if not root.is_dir() or root.is_symlink():
            continue
        for current, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(
                name for name in directories if allowed_name(name) and not (Path(current) / name).is_symlink()
            )
            candidates.extend(Path(current) / name for name in sorted(files) if Path(name).suffix in SOURCE_SUFFIXES)

    records = {}
    total_bytes = 0
    for path in sorted(candidates):
        if path.is_symlink() or not allowed_name(path.name) or not path.is_file():
            continue
        total_bytes += path.stat().st_size
        if len(records) >= MAX_SOURCE_FILES or total_bytes > MAX_SOURCE_BYTES:
            raise ValueError('source provenance exceeds file-count or byte limit; narrow the source tree')
        records[path.relative_to(source_root).as_posix()] = _file_record(path)
    return {
        'root': str(source_root),
        **_git_metadata(source_root),
        'mode': 'working-tree-sha256',
        'policy': {
            'directories': list(SOURCE_DIRECTORIES),
            'suffixes': sorted(SOURCE_SUFFIXES),
            'excluded_directories': sorted(EXCLUDED_DIRECTORIES),
            'hidden_files_and_directories': 'excluded',
            'symlinks': 'excluded',
            'secret_filename_pattern': SECRET_NAME.pattern,
            'max_files': MAX_SOURCE_FILES,
            'max_bytes': MAX_SOURCE_BYTES,
        },
        'files': records,
    }


def _dataset_records(config):
    roots = set()

    def visit(node):
        if isinstance(node, dict):
            if 'path_to_data_dir' in node and 'name' in node:
                roots.add((Path(node['path_to_data_dir']) / node['name']).resolve())
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(config['dataset'])
    if not roots:
        raise ValueError('cannot resolve dataset inputs for provenance: path_to_data_dir and name are required')
    return [
        {
            'root': str(root),
            'inputs': {name: _file_record(root / name) for name in DATASET_INPUTS},
            'preprocessing_manifests': {
                name: _file_record(root / name) for name in PREPROCESSING_MANIFESTS if (root / name).exists()
            },
        }
        for root in sorted(roots)
    ]


def _artifact_records(config):
    records = {}

    def visit(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if SECRET_NAME.search(key):
                    continue
                if isinstance(child, str) and key.startswith('path_to'):
                    if Path(child).is_file() or 'counts' in key or key == 'path_to_train_presence':
                        records[child] = _file_record(child)
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(config)
    return records


def _environment():
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'pandas', 'polars', 'tensorboard', 'irec'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    'index': index,
                    'name': properties.name,
                    'total_memory_bytes': properties.total_memory,
                    'compute_capability': [properties.major, properties.minor],
                }
            )
    environment_names = (
        'CUDA_VISIBLE_DEVICES',
        'CUBLAS_WORKSPACE_CONFIG',
        'PYTHONHASHSEED',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'POLARS_MAX_THREADS',
    )
    return {
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'packages': packages,
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'cuda_available': cuda_available,
        'gpus': gpus,
        'torch_num_threads': torch.get_num_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'environment_variables': {name: os.environ[name] for name in environment_names if name in os.environ},
    }


def write_report(path, report):
    """Write exclusively; pre-serialize so JSON errors cannot leave a report stub."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def begin_run(
    config,
    params_path,
    output_path,
    selection,
    experiment,
    source_root=None,
    evaluation_config=None,
    params_sha256=None,
    imported_modules=None,
):
    """Capture inputs and exclusively reserve report/manifest/checkpoint names before training."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', experiment) or experiment in ('.', '..'):
        raise ValueError('experiment name must be a safe filename for the checkpoint namespace')
    slugs = [name.replace('/', '_').replace('@', '_at_') for name in selection]
    if not slugs or len(set(slugs)) != len(slugs) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', slug) for slug in slugs):
        raise ValueError('selection metrics must have distinct safe checkpoint filenames')
    output = Path(os.path.abspath(output_path))
    manifest_path = Path(str(output) + '.provenance')
    checkpoints = Path.cwd() / 'checkpoints'
    reservation = checkpoints / f'{experiment}.provenance.json'
    collisions = [path for path in (output, manifest_path, reservation) if os.path.lexists(path)]
    if checkpoints.exists():
        collisions.extend(path for path in checkpoints.iterdir() if path.name.startswith(f'{experiment}_best_'))
    if collisions:
        raise FileExistsError(
            f'run artifacts already exist; choose a fresh output and checkpoint namespace: {collisions}'
        )

    source_root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    imported_sources = validate_imported_sources(source_root, imported_modules=imported_modules)
    started = datetime.now(timezone.utc).isoformat()
    params = _file_record(params_path)
    if params_sha256 is not None and params['sha256'] != params_sha256:
        raise RuntimeError('params changed after loading and before capturing provenance')
    source = _source_inventory(source_root)
    for name, origins in imported_sources.items():
        relative_path = Path(origins['file']).relative_to(source_root).as_posix()
        if relative_path not in source['files']:
            raise RuntimeError(
                f'Cannot verify imported source for {name}: {relative_path} is excluded from the inventory'
            )
    source['imported_modules'] = imported_sources
    manifest = {
        'schema_version': 1,
        'phase': 'before_training',
        'capture_started_at_utc': started,
        'working_directory': str(Path.cwd()),
        'report_path': str(output),
        'experiment': experiment,
        'checkpoint_namespace': str(reservation),
        'resolved_config': _redact(config),
        'evaluation_config': _redact(evaluation_config),
        'selection': list(selection),
        'seed': config['seed'],
        'step_limit': config['train_steps_num'],
        'params': params,
        'source': source,
        'datasets': _dataset_records(config),
        'artifacts': _artifact_records({'training': config, 'evaluation': evaluation_config}),
        'lockfiles': {name: _file_record(source_root / name) for name in LOCKFILES if (source_root / name).is_file()},
        'environment': _environment(),
    }
    manifest['captured_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_report(reservation, {'report_path': str(output), 'manifest_path': str(manifest_path)})
    try:
        write_report(manifest_path, manifest)
    except Exception:
        reservation.unlink()
        raise
    return {'manifest': manifest, 'path': str(manifest_path), 'sha256': file_sha256(manifest_path)}
