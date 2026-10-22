"""Independent CPU protocol oracles; no optimizer steps or scientific data writes."""

import copy
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import tarfile
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import irec.models.base as model_base
import irec.utils
from irec.callbacks.base import ValidationCallback
from irec.dataloader.batch_processors import BasicBatchProcessor
from irec.dataset.sequences import MCLSRDataset
from irec.metric.base import MCLSRHitRateMetric, MCLSRNDCGMetric, MCLSRRecallMetric
from irec.models import BaseModel


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = os.environ.get('MCLSR_SCREEN_EVIDENCE')


def _script(name):
    spec = importlib.util.spec_from_file_location(f'paper_protocol_{name}', ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(irec.utils, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(model_base, 'DEVICE', torch.device('cpu'))
    previous_threads = torch.get_num_threads()
    random_state = random.getstate()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(731)
            yield
    finally:
        random.setstate(random_state)
        torch.set_num_threads(previous_threads)


@pytest.fixture(scope='module')
def actual_screen():
    if not EVIDENCE:
        pytest.skip('Set MCLSR_SCREEN_EVIDENCE to the archived actual component screen')
    if not Path(EVIDENCE).is_file():
        pytest.fail('MCLSR_SCREEN_EVIDENCE must point to an existing archived component-screen file')
    with tarfile.open(EVIDENCE, 'r:gz') as bundle:
        prefix = 'component_screen_20261013/'
        manifest = json.load(bundle.extractfile(prefix + 'screen.json'))
        configs = {}
        for arm in manifest['configs']:
            payload = bundle.extractfile(prefix + f'configs/{arm}.json').read()
            assert hashlib.sha256(payload).hexdigest() == manifest['configs'][arm]['sha256']
            configs[arm] = json.loads(payload)
    return manifest, configs


def test_protocol_sources_and_config_are_the_actual_archived_screen(actual_screen):
    manifest, configs = actual_screen
    for filename in [
        'scripts/prepare_dataset.py',
        'scripts/generate_mclsr_role_counts.py',
        'src/irec/dataset/sequences.py',
        'src/irec/dataset/samplers/mclsr.py',
        'src/irec/dataloader/batch_processors.py',
        'src/irec/metric/base.py',
        'src/irec/models/base.py',
        'src/irec/models/mclsr.py',
        'src/irec/models/sasrec.py',
        'src/irec/callbacks/base.py',
        'src/irec/training.py',
    ]:
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == manifest['source_sha256'][filename]


def test_archived_screen_profile_matches_declared_experiment(actual_screen):
    manifest, configs = actual_screen
    full = configs['full_dq1']
    stripped = configs['downstream_only_dq1']
    assert full['model'] == stripped['model']
    assert full['model']['use_graph'] is True
    assert full['model']['alpha'] == 0.5
    assert full['model']['eval_top_k'] == 1000
    assert full['model']['embedding_dim'] == 64
    assert full['dataset']['dataset']['max_history_length'] == 20
    assert full['dataset']['dataset']['samplers']['num_negatives_train'] == 0
    validation = next(entry for entry in full['callback']['callbacks'] if entry['type'] == 'validation')
    assert validation['metrics'] == {
        'ndcg@20': {'k': 20, 'type': 'mclsr-ndcg'},
        'recall@1000': {'k': 1000, 'type': 'mclsr-recall'},
    }
    assert manifest['test_evaluation'] is False


def test_all_archived_screen_arms_isolate_declared_interventions(actual_screen):
    _, configs = actual_screen
    components = ('interest', 'user_feature', 'item_feature')
    removed_by_group = {'full': (), 'downstream_only': components}
    removed_by_group.update({f'no_{component}': (component,) for component in components})
    corrected_by_suffix = {'auxq0': (), 'all_q1': components}
    corrected_by_suffix.update({f'{component}_q1': (component,) for component in components})
    expected_names = {f'{group}_dq{coefficient}' for group in removed_by_group for coefficient in (0, 1)}
    expected_names.update(f'masked_dq1_{suffix}' for suffix in corrected_by_suffix)
    assert set(configs) == expected_names
    full = configs['full_dq0']
    for config in configs.values():
        for field in ('model', 'dataset', 'dataloader', 'optimizer'):
            assert config[field] == full[field]
        assert config['train_steps_num'] == 65536
        assert config['loss']['type'] == 'composite'
    for group, removed in removed_by_group.items():
        disabled = {f'contrastive_{component}_loss' for component in removed}
        for coefficient in (0, 1):
            expected = [copy.deepcopy(term) for term in full['loss']['losses'] if term['output_prefix'] not in disabled]
            for term in expected:
                if term['output_prefix'] == 'downstream_loss':
                    term['logq_lambda'] = float(coefficient)
            assert configs[f'{group}_dq{coefficient}']['loss']['losses'] == expected
    control = configs['masked_dq1_auxq0']['loss']['losses']
    assert control[0] == configs['full_dq1']['loss']['losses'][0]
    for term in control[1:]:
        assert term['type'] == 'fps_logq'
        assert term['scheme'] == 'paper'
        assert term['normalize_embeddings'] is True
        assert term['temperature'] == 0.5
        assert term['mask_false_negatives'] is True
        assert term['leave_own_out'] is False
        assert term['center_log_q'] is False
        assert term['logq_lambda'] == 0.0
    for suffix, corrected in corrected_by_suffix.items():
        expected = copy.deepcopy(control)
        for term in expected:
            if term['output_prefix'] in {f'contrastive_{component}_loss' for component in corrected}:
                term['logq_lambda'] = 1.0
        assert configs[f'masked_dq1_{suffix}']['loss']['losses'] == expected


def _metric_input(predictions, targets):
    return {
        'predictions': torch.tensor(predictions, dtype=torch.long),
        'labels.ids': torch.tensor([item for labels in targets for item in labels], dtype=torch.long),
        'labels.length': torch.tensor([len(labels) for labels in targets], dtype=torch.long),
    }


def _set_metric_oracle(predictions, targets, cutoff):
    relevant = set(targets)
    discounts = [1.0 / math.log2(rank + 2) for rank in range(cutoff)]
    hits = [item in relevant for item in predictions[:cutoff]]
    ideal = math.fsum(discounts[: min(cutoff, len(relevant))])
    return {
        'recall': sum(hits) / len(relevant) if relevant else 0.0,
        'ndcg': math.fsum(weight for weight, hit in zip(discounts, hits) if hit) / ideal if ideal else 0.0,
        'hit': float(any(hits)),
    }


@pytest.mark.parametrize('cutoff', [1, 20, 1000])
def test_metrics_against_set_oracle_with_large_target_sets_and_batch_partitions(cutoff):
    generator = random.Random(981)
    predictions = [generator.sample(range(1, 1011), 1000) for _ in range(6)]
    targets = [
        [],
        [predictions[1][0]],
        [predictions[2][19], predictions[2][20], predictions[2][999]],
        list(range(1, 1011)),
        list(range(2001, 2008)),
        [predictions[5][0], predictions[5][999], 2000],
    ]
    metrics = {'recall': MCLSRRecallMetric, 'ndcg': MCLSRNDCGMetric, 'hit': MCLSRHitRateMetric}
    expected = [_set_metric_oracle(prediction, labels, cutoff) for prediction, labels in zip(predictions, targets)]
    for name, metric_type in metrics.items():
        metric = metric_type(cutoff)
        reference = [row[name] for row in expected]
        assert metric(_metric_input(predictions, targets), 'predictions', 'labels') == pytest.approx(
            reference, abs=2e-6
        )
        for permutation in [list(range(6)), [5, 2, 0, 4, 1, 3]]:
            values = []
            for batch_indices in [permutation[:2], permutation[2:5], permutation[5:]]:
                batch = _metric_input(
                    [predictions[index] for index in batch_indices],
                    [targets[index] for index in batch_indices],
                )
                values.extend(metric(batch, 'predictions', 'labels'))
            assert values == pytest.approx([reference[index] for index in permutation], abs=2e-6)


def test_duplicate_target_events_use_set_relevance():
    predictions = [7, 9] + list(range(30, 48))
    targets = [7, 7, 9]
    inputs = _metric_input([predictions], [targets])
    set_oracle = _set_metric_oracle(predictions, targets, 20)
    actual_recall = MCLSRRecallMetric(20)(inputs, 'predictions', 'labels')[0]
    actual_ndcg = MCLSRNDCGMetric(20)(inputs, 'predictions', 'labels')[0]
    assert set_oracle['recall'] == set_oracle['ndcg'] == 1.0
    assert actual_recall == pytest.approx(set_oracle['recall'])
    assert actual_ndcg == pytest.approx(set_oracle['ndcg'])


def test_metric_rejects_padding_in_malformed_target_lists():
    inputs = _metric_input([[2] + list(range(30, 49))], [[0, 2]])
    for metric_type in (MCLSRRecallMetric, MCLSRNDCGMetric):
        with pytest.raises(ValueError, match='padding'):
            metric_type(20)(inputs, 'predictions', 'labels')


def test_duplicate_prediction_contract_is_enforced_by_metric():
    inputs = _metric_input([[7, 7] + list(range(30, 48))], [[7]])
    for metric_type in (MCLSRRecallMetric, MCLSRNDCGMetric):
        with pytest.raises(ValueError, match='unique'):
            metric_type(20)(inputs, 'predictions', 'labels')


def _inference_batch(histories, targets=None):
    if targets is None:
        targets = [[] for _ in histories]
    return BasicBatchProcessor()(
        [
            {
                'user.ids': [index + 1],
                'user.length': 1,
                'item.ids': history,
                'item.length': len(history),
                'labels.ids': labels,
                'labels.length': len(labels),
                'negatives.ids': [],
                'negatives.length': 0,
            }
            for index, (history, labels) in enumerate(zip(histories, targets))
        ]
    )


@pytest.fixture(params=[torch.float32, torch.float64], ids=['float32', 'float64'])
def inference_model(actual_screen, request):
    config = copy.deepcopy(actual_screen[1]['full_dq1']['model'])
    model = BaseModel.create_from_config(config, num_items=1003, num_users=4, max_sequence_length=20).to(
        device='cpu', dtype=request.param
    )
    generator = random.Random(20261017)
    table = [[generator.uniform(-0.8, 0.8) for _ in range(64)] for _ in range(1005)]
    table[1] = [1.0] + [0.0] * 63
    table[1003] = [10.0] + [0.0] * 63
    table[0] = [100000.0] + [0.0] * 63
    table[1004] = [200000.0] + [0.0] * 63
    with torch.no_grad():
        model._item_embeddings.weight.copy_(torch.tensor(table, dtype=request.param))
        for parameter in model._current_interest_learning_encoder.parameters():
            parameter.zero_()
    model.eval()
    return model, model._item_embeddings.weight.detach().tolist()


def _catalogue_oracle(table, history, cutoff=1000):
    query = [math.fsum(table[item][column] for item in history) / len(history) for column in range(len(table[0]))]
    scores = {
        item: math.fsum(first * second for first, second in zip(query, table[item]))
        for item in range(1, len(table) - 1)
    }
    ranking = sorted(scores, key=scores.get, reverse=True)
    assert all(abs(scores[first] - scores[second]) > 1e-10 for first, second in zip(ranking, ranking[1:]))
    return ranking[:cutoff]


@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_actual_eval_matches_full_catalogue_oracle_not_batch_candidates(inference_model, batch_size):
    model, table = inference_model
    histories = [[1], [4, 8, 12], list(range(20, 40))]
    expected = [_catalogue_oracle(table, history) for history in histories]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(histories), batch_size):
            predictions.extend(model(_inference_batch(histories[start : start + batch_size])).tolist())
    assert predictions == expected
    assert predictions[0][0] == 1003
    assert 1003 not in set(item for history in histories for item in history)
    assert 1 in predictions[0]
    assert all(len(set(ranking)) == 1000 and 0 not in ranking and 1004 not in ranking for ranking in predictions)
    excluded_item = _catalogue_oracle(table, histories[2], cutoff=1003)[-1]
    targets = [[1003, 1], [expected[1][0], expected[1][-1]], [expected[2][19], expected[2][20], excluded_item]]
    inputs = _metric_input(predictions, targets)
    for metric, name, cutoff in [(MCLSRNDCGMetric(20), 'ndcg', 20), (MCLSRRecallMetric(1000), 'recall', 1000)]:
        reference = [_set_metric_oracle(ranking, labels, cutoff)[name] for ranking, labels in zip(expected, targets)]
        assert metric(inputs, 'predictions', 'labels') == pytest.approx(reference, abs=2e-6)


def test_seen_items_are_retained_not_implicitly_filtered(inference_model):
    model, table = inference_model
    with torch.no_grad():
        ranking = model(_inference_batch([[1]])).tolist()[0]
    unfiltered = _catalogue_oracle(table, [1])
    filtered = [item for item in _catalogue_oracle(table, [1], cutoff=1003) if item != 1][:1000]
    assert ranking == unfiltered
    assert ranking != filtered
    assert 1 in ranking and 1 not in filtered


def test_actual_eval_skips_graph_and_projectors_and_does_not_cache_item_scores(inference_model, monkeypatch):
    model, _ = inference_model

    def forbidden(*arguments, **kwargs):
        pytest.fail('Evaluation called a graph encoder or contrastive projector')

    monkeypatch.setattr(model, '_apply_graph_encoder', forbidden)
    for projector in [
        model._sequential_projector,
        model._graph_projector,
        model._user_projection,
        model._item_projection,
    ]:
        monkeypatch.setattr(projector, 'forward', forbidden)
    batch = _inference_batch([[1]])
    with torch.no_grad():
        before = model(batch).clone()
        model._user_embeddings.weight.fill_(12345.0)
        model._alpha = -12345.0
        assert torch.equal(model(batch), before)
        model._item_embeddings.weight[999].zero_()
        model._item_embeddings.weight[999, 0] = 50.0
        after = model(batch)
        assert before[0, 0].item() == 1003
        assert after[0, 0].item() == 999
        assert torch.equal(after, model(batch))


def test_validation_macro_averages_users_and_characterizes_eval_state(inference_model, actual_screen, monkeypatch):
    model, table = inference_model
    excluded_target = _catalogue_oracle(table, [1], cutoff=1003)[-1]
    batches = [_inference_batch([[1], [1]], [[1003], [1003]]), _inference_batch([[1]], [[excluded_target]])]
    states = []
    handle = model.register_forward_pre_hook(
        lambda module, inputs: states.append((module.training, torch.is_grad_enabled()))
    )
    writer = SimpleNamespace(add_scalar=lambda *arguments: None, flush=lambda: None)
    monkeypatch.setattr(irec.utils.tensorboards, 'GLOBAL_TENSORBOARD_WRITER', writer)
    config = next(
        entry for entry in actual_screen[1]['full_dq1']['callback']['callbacks'] if entry['type'] == 'validation'
    )
    callback = ValidationCallback.create_from_config(
        config,
        model=model,
        train_dataloader=None,
        validation_dataloader=batches,
        eval_dataloader=None,
        optimizer=None,
    )
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.train()
    output = {}
    callback(output, 0)
    handle.remove()
    assert output['validation/ndcg@20'] == pytest.approx(2 / 3)
    assert output['validation/recall@1000'] == pytest.approx(2 / 3)
    assert output['validation/recall@1000'] != pytest.approx((1 + 0) / 2)
    assert states == [(False, False), (False, False)]
    assert model.training is False
    assert all(
        torch.equal(parameter, original[name]) and parameter.grad is None
        for name, parameter in model.named_parameters()
    )
    model.train()
    assert model.training and model._dropout.training


def test_no_graph_and_full_graph_flags_have_same_eval_with_shared_weights(inference_model, actual_screen):
    full_model, _ = inference_model
    config = copy.deepcopy(actual_screen[1]['full_dq1']['model'])
    config['use_graph'] = False
    no_graph = BaseModel.create_from_config(config, num_items=1003, num_users=4, max_sequence_length=20).to(
        device='cpu', dtype=full_model._item_embeddings.weight.dtype
    )
    no_graph.load_state_dict({name: full_model.state_dict()[name] for name in no_graph.state_dict()})
    no_graph.eval()
    with torch.no_grad():
        batch = _inference_batch([[1], [4, 8, 12]])
        assert torch.equal(no_graph(batch), full_model(batch))


def test_stripped_mclsr_is_not_sasrec_a_constructive_ranking_counterexample(actual_screen):
    config = copy.deepcopy(actual_screen[1]['full_dq1']['model'])
    config.update(use_graph=False, eval_top_k=1)
    mclsr = BaseModel.create_from_config(config, num_items=3, num_users=1, max_sequence_length=20).double().cpu()
    sasrec = (
        BaseModel.create_from_config(
            {
                'type': 'sasrec_in_batch',
                'sequence_prefix': 'item',
                'positive_prefix': 'positive',
                'embedding_dim': 64,
                'num_heads': 2,
                'num_layers': 1,
                'dim_feedforward': 256,
                'layer_norm_eps': config['layer_norm_eps'],
                'dropout': config['dropout'],
                'eval_top_k': 1,
                'query_positions': 'last',
            },
            num_items=3,
            max_sequence_length=20,
        )
        .double()
        .cpu()
    )
    table = torch.zeros(5, 64, dtype=torch.float64)
    table[1, 0] = 2.0
    table[2] = 3.0
    with torch.no_grad():
        for model in [mclsr, sasrec]:
            model._item_embeddings.weight.copy_(table)
            model._position_embeddings.weight.zero_()
        for name, parameter in sasrec._encoder.named_parameters():
            parameter.fill_(1.0 if 'norm' in name and 'weight' in name else 0.0)
        mclsr.eval()
        sasrec.eval()
        batch = _inference_batch([[1]])
        assert mclsr(batch).tolist() == [[2]]
        assert sasrec(batch).tolist() == [[1]]
    raw_scores = [4.0, 6.0, 0.0]
    mean = 2 / 64
    variance = ((2 - mean) ** 2 + 63 * mean**2) / 64
    normalized = [(2 - mean) / math.sqrt(variance), *([-mean / math.sqrt(variance)] * 63)]
    normalized_scores = [2 * normalized[0], 3 * math.fsum(normalized), 0.0]
    assert max(range(3), key=raw_scores.__getitem__) + 1 == 2
    assert max(range(3), key=normalized_scores.__getitem__) + 1 == 1


def _five_core_oracle(events):
    retained = list(events)
    stages = 0
    while True:
        users = Counter(user for user, item, rating, timestamp in retained)
        items = Counter(item for user, item, rating, timestamp in retained)
        filtered = [event for event in retained if users[event[0]] >= 5 and items[event[1]] >= 5]
        stages += 1
        if len(filtered) == len(retained):
            break
        retained = filtered
    user_ids = {user: index + 1 for index, user in enumerate(sorted({event[0] for event in retained}))}
    item_ids = {item: index + 1 for index, item in enumerate(sorted({event[1] for event in retained}))}
    sequences = []
    for user in user_ids:
        chronological = sorted((event for event in retained if event[0] == user), key=lambda event: event[3])
        sequences.append((user_ids[user], [item_ids[event[1]] for event in chronological]))
    return sequences, stages


def test_iterative_user_item_filter_differs_from_item_only_and_preserves_low_ratings(tmp_path):
    prepare = _script('prepare_dataset')
    events = [
        (f'u{user}', f'item{item}', 1.0, 1000 - 7 * item + user)
        for user in [10, 30, 50, 70, 90]
        for item in [11, 22, 33, 44, 55]
    ]
    events.extend(
        (user, 'fragile', 1.0, 900 + index) for index, user in enumerate(['u10', 'weak1', 'weak2', 'weak3', 'weak4'])
    )
    filename = tmp_path / 'ratings.csv'
    filename.write_text(''.join(','.join(map(str, event)) + '\n' for event in reversed(events)))
    expected, stages = _five_core_oracle(events)
    assert stages >= 3
    assert prepare.sequences_from_csv(filename) == expected
    assert sum(map(lambda record: len(record[1]), expected)) == 25
    item_only = Counter(event[1] for event in events)
    assert sum(item_only[event[1]] >= 5 for event in events) == 30


def test_filter_counts_interaction_events_not_distinct_neighbors(tmp_path):
    prepare = _script('prepare_dataset')
    events = [('same-user', 'same-item', 1.0, timestamp) for timestamp in range(5)]
    filename = tmp_path / 'duplicates.csv'
    filename.write_text(''.join(','.join(map(str, event)) + '\n' for event in events))
    expected, _ = _five_core_oracle(events)
    assert expected == [(1, [1, 1, 1, 1, 1])]
    assert prepare.sequences_from_csv(filename) == expected


def test_user_split_is_seeded_on_sparse_user_ids_not_sequence_lengths():
    prepare = _script('prepare_dataset')
    sequences = [(user, list(range(user // 10 + 5))) for user in range(10, 101, 10)]
    expected = ({80, 40, 30, 90, 60, 70, 100, 50}, {10}, {20})
    assert prepare.split_users(sequences) == expected
    assert prepare.split_users([(user, items * 17) for user, items in sequences]) == expected
    assert len(set.union(*expected)) == 10
    assert not any(
        first & second
        for first, second in [(expected[0], expected[1]), (expected[0], expected[2]), (expected[1], expected[2])]
    )


@pytest.mark.parametrize('protocol', ['windowed-v2', 'legacy-v1'])
def test_streaming_prefix_oracle_and_raw_target_counts(protocol, tmp_path, monkeypatch, actual_screen):
    prepare = _script('prepare_dataset')
    sequences = [
        (10, [0, 7, 0, 8]),
        (30, list(range(100, 127))),
        (50, [8, 7, 8, 0, 7, 8]),
        (70, list(range(200, 230))),
        (90, [0, 7, 0, 7, 8]),
    ]
    train_users, valid_users, test_users = {10, 30, 50}, {70}, {90}
    directory = tmp_path / protocol
    prepare.write_files(sequences, train_users, valid_users, test_users, directory, protocol=protocol)
    user_map = {user: index + 1 for index, (user, _) in enumerate(sequences)}
    item_map = {
        item: index + 1 for index, item in enumerate(sorted({item for _, items in sequences for item in items}))
    }
    config = copy.deepcopy(actual_screen[1]['full_dq1']['dataset']['dataset'])
    config.update(path_to_data_dir=str(tmp_path), name=protocol)
    if protocol == 'legacy-v1':
        config.pop('max_history_length')
    dataset = MCLSRDataset.create_from_config(config)
    train, validation, test = dataset.get_samplers()
    expected = []
    raw_targets = []
    for user, items in sequences:
        if user not in train_users:
            continue
        selected = items if protocol == 'windowed-v2' else items[-20:]
        bounded_history = deque(maxlen=20 if protocol == 'windowed-v2' else 19)
        for item in selected:
            mapped = item_map[item] if protocol == 'windowed-v2' else item
            if bounded_history:
                expected.append((user_map[user] if protocol == 'windowed-v2' else user, list(bounded_history), mapped))
                raw_targets.append(mapped)
            bounded_history.append(mapped)
    observed = [
        (train[index]['user.ids'][0], train[index]['item.ids'], train[index]['labels.ids'][0])
        for index in range(len(train))
    ]
    assert observed == expected
    assert len(train) == (34 if protocol == 'windowed-v2' else 27)
    assert max(len(history) for _, history, _ in observed) == (20 if protocol == 'windowed-v2' else 19)
    assert all(train[index]['negatives.ids'] == [] for index in range(len(train)))
    valid_source = sequences[3][1]
    boundary = 4 * len(valid_source) // 5
    transform = item_map.__getitem__ if protocol == 'windowed-v2' else lambda item: item
    assert validation[0]['item.ids'] == list(map(transform, valid_source[:boundary][-20:]))
    assert validation[0]['labels.ids'] == list(map(transform, valid_source[boundary:]))
    assert validation[0]['labels.length'] == 6
    assert test[0]['labels.ids'] == [transform(8)]
    if protocol == 'windowed-v2':
        assert all(item > 0 for _, history, target in observed for item in [*history, target])
    target_path = tmp_path / f'{protocol}_target.pkl'
    context_path = tmp_path / f'{protocol}_context.pkl'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'generate_mclsr_role_counts.py',
            '--input',
            str(directory / 'train_mclsr.txt'),
            '--num_items',
            str(dataset.num_items),
            '--target_output',
            str(target_path),
            '--context_output',
            str(context_path),
        ],
    )
    _script('generate_mclsr_role_counts').main()
    with target_path.open('rb') as stream:
        artifact = pickle.load(stream)
    counts = Counter(raw_targets)
    assert artifact['counts'].tolist() == [counts[index] for index in range(dataset.num_items + 2)]
    assert artifact['denominator'] == artifact['num_lines'] == len(raw_targets)
    assert artifact['role'] == 'target'
    assert artifact['max_len'] == (21 if protocol == 'windowed-v2' else 20)
    assert artifact['source_sha256'] == hashlib.sha256((directory / 'train_mclsr.txt').read_bytes()).hexdigest()
