"""Tiny float32 forward/backward parity check, not a training or paper-replication run.

Usage: python scripts/check_device_parity.py --device cuda:0 --report parity.json
The default --suite base retains MCLSR/SASRec with unmasked auxiliary FpsLoss.
Use --suite logq for MCLSR masked auxiliary Q0, IL-only, UC-only, IC-only and
all-Q1 cases (downstream Q1 in every case). All five share initialization, batch,
graphs and weights; only auxiliary logq_lambda changes. IL/UC use user-row Q in
sample mode; IC uses context-line inclusion Q with four line draws, not the
number of unique items/events. The production model deduplicates repeated IC
item IDs. These synthetic proposal models are not exact post-mask probabilities.
Each case reports source hashes, constructor/config/count provenance and scalar
component values. No training, optimizer steps or real dataset are involved.
Use --device cpu for a self-check. Floating tensors must satisfy the elementwise
bound abs(target-reference) <= atol + rtol*abs(reference); integer IDs must match
exactly. Reported max relative errors use max(abs(reference), 1e-12) as denominator.
"""

import argparse
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import irec.utils as utils  # noqa: E402
import irec.models.base as model_base  # noqa: E402
from irec.loss import FpsLoss, FpsLogQLoss, InBatchLogQLoss, load_counts_artifact  # noqa: E402
from irec.models import BaseModel  # noqa: E402


LOGQ_VARIANTS = {
    'mclsr_q0_masked': (),
    'mclsr_logq_il': ('L_IL',),
    'mclsr_logq_uc': ('L_UC',),
    'mclsr_logq_ic': ('L_IC',),
    'mclsr_logq_all': ('L_IL', 'L_UC', 'L_IC'),
}


@contextmanager
def device_scope(device):
    """Route legacy allocation helpers to the explicit device without editing model code."""
    previous = utils.DEVICE, model_base.DEVICE
    utils.DEVICE = model_base.DEVICE = device
    try:
        yield
    finally:
        utils.DEVICE, model_base.DEVICE = previous


def compare_tensors(reference, target, *, atol, rtol):
    result = {'passed': False, 'finite': False, 'max_abs_error': None, 'max_rel_error': None}
    if reference is None or target is None:
        return dict(result, reason='Missing tensor or gradient')
    if reference.shape != target.shape or reference.dtype != target.dtype:
        return dict(result, reason='Shape or dtype mismatch')
    reference = reference.detach().cpu()
    target = target.detach().cpu()
    if not (torch.isfinite(reference).all() and torch.isfinite(target).all()):
        return dict(result, reason='Nonfinite reference or target')
    difference = (target.double() - reference.double()).abs()
    relative = difference / reference.double().abs().clamp_min(1e-12)
    if reference.is_floating_point():
        passed = bool((difference <= atol + rtol * reference.double().abs()).all())
    else:
        passed = torch.equal(reference, target)
    return dict(
        result,
        passed=passed,
        finite=True,
        max_abs_error=float(difference.max()) if difference.numel() else 0.0,
        max_rel_error=float(relative.max()) if relative.numel() else 0.0,
    )


def fixed_graphs():
    interactions = torch.zeros(6, 14)
    for user_id, item_ids in (
        (1, [1, 3, 5, 7, 9]),
        (2, [2, 4, 6, 8]),
        (3, [6, 7, 9, 11, 12]),
        (4, [1, 2, 6, 12]),
    ):
        interactions[user_id, item_ids] = 1.0
    common = torch.zeros(20, 20)
    common[:6, 6:] = interactions
    common[6:, :6] = interactions.T
    user = interactions @ interactions.T
    item = interactions.T @ interactions
    user.fill_diagonal_(0)
    item.fill_diagonal_(0)
    graphs = {}
    for name, adjacency in (('graph', common), ('user_graph', user), ('item_graph', item)):
        inverse_degree = adjacency.sum(dim=1).clamp_min(1).rsqrt()
        graphs[name] = (inverse_degree[:, None] * adjacency * inverse_degree[None, :]).to_sparse().coalesce()
    return graphs


def fixed_case(name, counts_path):
    return build_case(name, counts_path)[0]


def role_count_fixtures(batch, directory):
    """Count users, targets and per-line context presence in the four synthetic rows."""
    context_counts = torch.zeros(14, dtype=torch.long)
    for context in torch.split(batch['item.ids'], batch['item.length'].tolist()):
        context_counts[context.unique()] += 1
    counts_by_role = {
        'user-row': torch.bincount(batch['user.ids'], minlength=6),
        'target': torch.bincount(batch['labels.ids'], minlength=14),
        'context-inclusion': context_counts,
    }
    paths = {}
    for role, counts in counts_by_role.items():
        paths[role] = Path(directory) / f'synthetic_{role}.pkl'
        artifact = {
            'counts': counts.numpy(),
            'denominator': batch['user.ids'].numel(),
            'role': role,
            'max_len': 5,
            'source': 'check_device_parity synthetic batch rows; no external data',
        }
        paths[role].write_bytes(pickle.dumps(artifact))
    return paths


def build_case(name, counts_path, *, variant=None):
    batch = {
        'item.ids': torch.tensor([1, 3, 5, 7, 2, 4, 8, 1, 3, 6, 7, 9, 11]),
        'item.length': torch.tensor([4, 3, 2, 4]),
        'user.ids': torch.tensor([1, 2, 1, 3]),
        'labels.ids': torch.tensor([9, 6, 9, 12]),
        'negatives.ids': torch.empty(0, dtype=torch.long),
        'positive.ids': torch.tensor([3, 5, 7, 9, 4, 8, 6, 3, 9, 7, 9, 11, 12]),
    }
    config = {
        'sequence_prefix': 'item',
        'embedding_dim': 16,
        'dropout': 0.0,
        'eval_top_k': 3,
        'layer_norm_eps': 1e-9,
    }
    meta = {'num_items': 12, 'num_users': 4, 'max_sequence_length': 5}
    if name == 'mclsr':
        config.update(
            type='mclsr',
            user_prefix='user',
            labels_prefix='labels',
            use_graph=True,
            graph_dropout=0.0,
            num_graph_layers=2,
            share_il_projector=True,
        )
        meta.update(fixed_graphs())
        query_prefix, positive_prefix = 'combined_representation', 'label_representation'
    else:
        config.update(
            type='sasrec_in_batch',
            positive_prefix='positive',
            num_heads=2,
            num_layers=2,
            dim_feedforward=32,
            query_positions='all',
        )
        query_prefix, positive_prefix = 'query_embeddings', 'positive_embeddings'
    model = BaseModel.create_from_config(config, **meta)
    losses = torch.nn.ModuleDict()
    loss_configs = {}
    count_artifacts = {}

    def add_loss(loss_name, loss_type, **kwargs):
        arguments = inspect.signature(loss_type).bind(**kwargs)
        arguments.apply_defaults()
        losses[loss_name] = loss_type(**kwargs)
        loss_configs[loss_name] = {
            'class': f'{loss_type.__module__}.{loss_type.__name__}',
            'kwargs': {
                key: value.name if isinstance(value, Path) else value for key, value in arguments.arguments.items()
            },
        }
        for key in ('path_to_counts', 'path_to_item_counts'):
            if key in kwargs:
                path = Path(kwargs[key])
                counts, metadata = load_counts_artifact(path)
                count_artifacts[path.name] = {
                    **metadata,
                    'counts': torch.as_tensor(counts).tolist(),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                }

    role_paths = role_count_fixtures(batch, Path(counts_path).parent) if variant is not None else {}
    add_loss(
        'L_P',
        InBatchLogQLoss,
        queries_prefix=query_prefix,
        positive_prefix=positive_prefix,
        positive_ids_prefix='positive_ids',
        user_ids_prefix='user_ids',
        path_to_item_counts=role_paths.get('target', counts_path),
        logq_lambda=1.0,
        **({'expected_counts_role': 'target', 'expected_counts_max_len': 5} if variant is not None else {}),
    )
    weights = {'L_P': 1.0}
    if name == 'mclsr':
        for loss_name, first, second, weight in (
            ('L_IL', 'sequential_representation', 'graph_representation', 1.0),
            ('L_UC', 'common_graph_user_embeddings', 'user_graph_user_embeddings', 0.05),
            ('L_IC', 'common_graph_item_embeddings', 'item_graph_item_embeddings', 0.05),
        ):
            kwargs = dict(
                fst_embeddings_prefix=first,
                snd_embeddings_prefix=second,
                tau=0.5,
                scheme='paper',
                normalize_embeddings=True,
            )
            if variant is not None:
                is_item = loss_name == 'L_IC'
                role = 'context-inclusion' if is_item else 'user-row'
                kwargs.update(
                    ids_prefix='item_ids' if is_item else 'user_ids',
                    path_to_counts=role_paths[role],
                    expected_counts_role=role,
                    expected_counts_max_len=5,
                    logq_lambda=float(loss_name in LOGQ_VARIANTS[variant]),
                    logq_probability_mode='inbatch_negative' if is_item else 'sample',
                    num_draws_value=batch['user.ids'].numel() if is_item else None,
                    mask_false_negatives=True,
                    leave_own_out=False,
                    center_log_q=False,
                )
            add_loss(loss_name, FpsLogQLoss if variant is not None else FpsLoss, **kwargs)
            weights[loss_name] = weight
    configuration = {
        'model': config,
        'metadata': {
            key: {'shape': list(value.shape), 'indices': value.indices().tolist(), 'values': value.values().tolist()}
            if isinstance(value, torch.Tensor)
            else value
            for key, value in meta.items()
        },
        'batch': {key: value.tolist() for key, value in batch.items()},
        'losses': loss_configs,
        'weights': weights,
        'count_artifacts': count_artifacts,
    }
    sources = {Path(__file__).resolve(), Path(utils.__file__).resolve()}
    for subject in (
        type(model),
        model_base.TorchModel,
        BaseModel.create_from_config,
        utils.create_masked_tensor,
        load_counts_artifact,
        *[type(loss) for loss in losses.values()],
    ):
        sources.add(Path(inspect.getsourcefile(subject)).resolve())
    root = Path(__file__).resolve().parents[1]
    provenance = {
        'config': configuration,
        'config_sha256': hashlib.sha256(
            json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        'source_sha256': {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)
        },
    }
    return (model, losses, weights, batch), provenance


def run_once(model, losses, weights, batch, device):
    """Copy one initialized CPU case; evaluate one total-loss backward without an optimizer."""
    with device_scope(device):
        model = copy.deepcopy(model).to(device).train()
        losses = copy.deepcopy(losses).to(device)
        for attribute in ('_graph', '_user_graph', '_item_graph'):
            graph = getattr(model, attribute, None)
            if graph is not None:
                setattr(model, attribute, graph.to(device))
        outputs = model({name: value.to(device) for name, value in batch.items()})
        loss_values = {name: loss(outputs) for name, loss in losses.items()}
        total = sum(weights[name] * value for name, value in loss_values.items())
        total.backward()
        tensors = {f'output/{name}': value for name, value in outputs.items()}
        tensors.update({f'loss/{name}': value for name, value in loss_values.items()})
        tensors['loss/total'] = total
        tensors.update({f'gradient/{name}': parameter.grad for name, parameter in model.named_parameters()})
        for name, value in tensors.items():
            if value is not None and value.device != device:
                raise RuntimeError(f'{name} is on {value.device}, expected {device}')
        return {name: value.detach().cpu().clone() if value is not None else None for name, value in tensors.items()}


def run_parity(device, *, seed=20261011, atol=1e-4, rtol=1e-3, suite='base'):
    if suite not in ('base', 'logq'):
        raise ValueError('Suite must be base or logq')
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError('Tolerances must be finite and nonnegative')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Target device must be cpu or cuda[:index]')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; requested comparison was not run')
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        torch.empty(0, device=device)
    else:
        device = torch.device('cpu')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        'passed': True,
        'suite': suite,
        'reference_device': 'cpu',
        'target_device': str(device),
        'seed': seed,
        'atol': atol,
        'rtol': rtol,
        'dtype': 'float32',
        'cpu_threads': 1,
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'target_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'tf32': False,
        'dropout': 0.0,
        'graph_dropout': 0.0,
        'scope': 'One synthetic forward/backward per case per device; no optimizer or real data',
        'contrastive': {'scheme': 'paper', 'similarity': 'cosine', 'tau': 0.5, 'logq': suite == 'logq'},
        'retrieval': {'logq_lambda': 1.0, 'variant': 'negatives_only', 'same_user_mask': True},
        'legacy_device_globals_overridden': ['irec.utils.DEVICE', 'irec.models.base.DEVICE'],
        'models': {},
    }
    with tempfile.TemporaryDirectory(prefix='mclsr-parity-') as directory:
        counts_path = Path(directory) / 'synthetic_counts.pkl'
        with counts_path.open('wb') as output:
            pickle.dump(torch.arange(1, 15, dtype=torch.float32).square().numpy(), output)
        for name in ('mclsr', 'sasrec') if suite == 'base' else LOGQ_VARIANTS:
            if suite == 'logq':
                torch.manual_seed(seed)
            case, provenance = build_case(
                name if suite == 'base' else 'mclsr',
                counts_path,
                variant=name if suite == 'logq' else None,
            )
            reference = run_once(*case, torch.device('cpu'))
            target = run_once(*case, device)
            checks = [
                dict(name=key, **compare_tensors(reference.get(key), target.get(key), atol=atol, rtol=rtol))
                for key in sorted(reference.keys() | target.keys())
            ]
            passed = bool(checks) and all(check['passed'] for check in checks)
            components = {}
            for key in sorted(reference.keys() | target.keys()):
                if key.startswith('loss/'):
                    components[key.removeprefix('loss/')] = {
                        side: float(snapshot[key])
                        if snapshot.get(key) is not None and torch.isfinite(snapshot[key]).all()
                        else None
                        for side, snapshot in (('reference', reference), ('target', target))
                    }
            report['models'][name] = {
                'passed': passed,
                'checks': checks,
                'components': components,
                'provenance': provenance,
            }
            report['passed'] &= passed
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--suite',
        choices=('base', 'logq'),
        default='base',
        help='base: unchanged models; logq: five matched MCLSR auxiliary cases',
    )
    parser.add_argument('--report', type=Path)
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-3)
    args = parser.parse_args(argv)
    try:
        report = run_parity(args.device, seed=args.seed, atol=args.atol, rtol=args.rtol, suite=args.suite)
    except (RuntimeError, ValueError, OSError) as error:
        report = {'passed': False, 'target_device': args.device, 'error': str(error)}
    if args.report:
        try:
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        except OSError as error:
            print(f'FAIL writing report: {error}', file=sys.stderr)
            return 1
    if 'error' in report:
        print(f'FAIL: {report["error"]}', file=sys.stderr)
    for name, result in report.get('models', {}).items():
        checks = result['checks']
        max_abs = max(check['max_abs_error'] or 0.0 for check in checks)
        max_rel = max(check['max_rel_error'] or 0.0 for check in checks)
        print(
            f'{"PASS" if result["passed"] else "FAIL"} {name}: {len(checks)} checks; max_abs={max_abs:.3e}, max_rel={max_rel:.3e}'
        )
        for check in checks:
            if not check['passed']:
                print(f'  FAIL {check["name"]}: {json.dumps(check, allow_nan=False)}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
