"""Independent NumPy/finite-difference audit of MCLSR equations 1-13 and 15.

These are small CPU forward/backward checks, not optimization or training runs.
Graph cache I/O is disabled; no scientific data, checkpoints or configs are written.
The supervised term in the composition test is an explicitly synthetic probe.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sparse
import torch

import irec.dataset.graph as graph_module
import irec.utils as utility_module
from irec.dataset.graph import GraphDataset
from irec.loss.base import CompositeLoss
from irec.loss.contrastive import FpsLoss
from irec.models.mclsr import MCLSRModel


PROFILES = ('paper_core', 'regularized', 'repeated_small_domain')
VIEW_PAIRS = (
    ('sequential_representation', 'graph_representation'),
    ('common_graph_user_embeddings', 'user_graph_user_embeddings'),
    ('common_graph_item_embeddings', 'item_graph_item_embeddings'),
)
HEAD_OUTPUTS = tuple(name for pair in VIEW_PAIRS for name in pair)
RAW_OUTPUTS = ('raw_sequential', 'raw_general', 'combined_representation')


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    monkeypatch.setattr(utility_module, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(graph_module, 'DEVICE', torch.device('cpu'))


def _emit(name, **values):
    print('ARCH_ORACLE ' + json.dumps({'case': name, **values}, sort_keys=True))


def _normalize_adjacency(adjacency):
    normalized = np.zeros_like(adjacency, dtype=np.float64)
    degrees = [sum(row) for row in adjacency]
    for receiver in range(len(adjacency)):
        for sender in range(len(adjacency)):
            if degrees[receiver] and degrees[sender]:
                normalized[receiver, sender] = adjacency[receiver, sender] / np.sqrt(
                    degrees[receiver] * degrees[sender]
                )
    return normalized


def _propagate(operator, features, depth=2, layer_mean=False):
    states = [features]
    for layer in range(1, depth + 1):
        states.append(np.linalg.matrix_power(operator, layer) @ features)
    return sum(states) / len(states) if layer_mean else states[-1]


def _softmax(logits):
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


def _project(features, parameters, prefix):
    hidden = features @ parameters[prefix + '.0.weight'].T + parameters[prefix + '.0.bias']
    activated = np.where(hidden >= 0, hidden, np.expm1(hidden))
    return activated @ parameters[prefix + '.2.weight'].T + parameters[prefix + '.2.bias']


def _paper_contrastive(first, second, mean=True, entity_ids=None):
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-6)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-6)
    terms = []
    for anchor in range(len(first)):
        positive = float(first[anchor] @ second[anchor]) / 0.5
        scores = [positive]
        for candidate in range(len(first)):
            if candidate == anchor:
                continue
            if entity_ids is not None and entity_ids[candidate] == entity_ids[anchor]:
                continue
            scores.append(float(first[anchor] @ second[candidate]) / 0.5)
            scores.append(float(first[anchor] @ first[candidate]) / 0.5)
        maximum = max(scores)
        terms.append(maximum + np.log(sum(np.exp(score - maximum) for score in scores)) - positive)
    return float(np.mean(terms) if mean else sum(terms))


def _probe(features):
    coefficients = np.sin(np.arange(features.size).reshape(features.shape) + 0.4)
    return float(np.sum(coefficients * features + 0.17 * features**2) / features.size)


class _SyntheticPredictionProbe(torch.nn.Module):
    def forward(self, inputs):
        features = inputs['combined_representation']
        coefficients = torch.sin(torch.arange(features.numel(), dtype=features.dtype).reshape(features.shape) + 0.4)
        return (coefficients * features + 0.17 * features.square()).sum() / features.numel()


def _auxiliary_losses(mean=True):
    return [
        FpsLoss(first, second, tau=0.5, normalize_embeddings=True, use_mean=mean, scheme='paper')
        for first, second in VIEW_PAIRS
    ]


def _objective(outputs, probe):
    if probe == 'joint':
        return _probe(outputs['combined_representation']) + sum(
            weight * _paper_contrastive(outputs[first], outputs[second])
            for weight, (first, second) in zip((1.0, 0.05, 0.05), VIEW_PAIRS)
        )
    selected = RAW_OUTPUTS if probe == 'readouts' else HEAD_OUTPUTS
    return sum((index + 1) * _probe(outputs[name]) for index, name in enumerate(selected))


def _torch_objective(outputs, probe):
    if probe == 'joint':
        return CompositeLoss([_SyntheticPredictionProbe(), *_auxiliary_losses()], [1.0, 1.0, 0.05, 0.05])(outputs)
    selected = RAW_OUTPUTS if probe == 'readouts' else HEAD_OUTPUTS
    result = outputs[selected[0]].new_zeros(())
    for index, name in enumerate(selected):
        features = outputs[name]
        coefficients = torch.sin(torch.arange(features.numel(), dtype=features.dtype).reshape(features.shape) + 0.4)
        result = result + (index + 1) * (coefficients * features + 0.17 * features.square()).sum() / features.numel()
    return result


def _finite_differences(function, parameters, step=1e-6):
    gradients = {}
    for name, values in parameters.items():
        gradient = np.zeros_like(values)
        for coordinate in np.ndindex(values.shape):
            original = values[coordinate]
            values[coordinate] = original + step
            upper = function(parameters)
            values[coordinate] = original - step
            lower = function(parameters)
            values[coordinate] = original
            gradient[coordinate] = (upper - lower) / (2 * step)
        gradients[name] = gradient
    return gradients


def _compare_gradients(actual, parameters, reference):
    tensors = dict(parameters)
    snapshots = {name: parameter.detach().numpy().copy() for name, parameter in tensors.items()}
    observed = torch.autograd.grad(actual, tuple(tensors.values()), allow_unused=True)
    expected = _finite_differences(reference, snapshots)
    maximum_error = 0.0
    tested_coordinates = 0
    for (name, parameter), gradient in zip(tensors.items(), observed):
        value = np.zeros(parameter.shape) if gradient is None else gradient.detach().numpy()
        np.testing.assert_allclose(value, expected[name], atol=4e-7, rtol=5e-5, err_msg=name)
        maximum_error = max(maximum_error, float(np.max(np.abs(value - expected[name]))))
        tested_coordinates += parameter.numel()
    return maximum_error, tested_coordinates


def _operators():
    incidence = np.zeros((5, 7))
    incidence[1:4, 1:6] = [[1, 2, 0, 1, 0], [0, 1, 3, 0, 2], [2, 0, 1, 2, 1]]
    common = np.block([[np.zeros((5, 5)), incidence], [incidence.T, np.zeros((7, 7))]])
    users = np.zeros((5, 5))
    users[1:4, 1:4] = [[0, 5, 0], [5, 0, 0], [0, 3, 0]]
    items = np.zeros((7, 7))
    items[1:6, 1:6] = [[0, 2, 0, 1, 0], [1, 0, 3, 0, 1], [0, 2, 0, 4, 0], [1, 0, 1, 0, 2], [0, 3, 1, 0, 0]]
    return tuple(_normalize_adjacency(adjacency) for adjacency in (common, users, items))


def _fixture(profile='paper_core', shared=True, histories=None, alpha=0.5):
    regularized = profile != 'paper_core'
    repeated = profile == 'repeated_small_domain'
    if histories is None:
        histories = ((1, 2, 1), (2,), (1, 2)) if repeated else ((1, 2, 1), (2, 4), (3, 5, 2))
    user_ids = (1, 1, 3) if repeated else (1, 2, 3)
    operators = _operators()
    model = (
        MCLSRModel(
            sequence_prefix='item',
            user_prefix='user',
            labels_prefix='labels',
            negatives_prefix='negatives',
            num_users=3,
            num_items=5,
            max_sequence_length=3,
            embedding_dim=3,
            num_graph_layers=2,
            common_graph=torch.tensor(operators[0], dtype=torch.float64).to_sparse().coalesce(),
            user_graph=torch.tensor(operators[1], dtype=torch.float64).to_sparse().coalesce(),
            item_graph=torch.tensor(operators[2], dtype=torch.float64).to_sparse().coalesce(),
            dropout=0.3 if regularized else 0.0,
            graph_dropout=0.3 if regularized else 0.0,
            layer_norm_eps=1e-9,
            alpha=alpha,
            share_il_projector=shared,
            use_layernorm=regularized,
        )
        .double()
        .train()
    )
    generator = np.random.default_rng(613)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = generator.uniform(-0.65, 0.65, parameter.shape)
            if name == '_layernorm.weight':
                values += 1.0
            parameter.copy_(torch.from_numpy(values))
    batch = {
        'item.ids': torch.tensor([item for sequence in histories for item in sequence]),
        'item.length': torch.tensor([len(sequence) for sequence in histories]),
        'user.ids': torch.tensor(user_ids),
        'labels.ids': torch.tensor([5, 3, 4]),
        'negatives.ids': torch.empty(0, dtype=torch.long),
    }
    context = {
        'histories': histories,
        'user_ids': user_ids,
        'operators': operators,
        'regularized': regularized,
        'shared': shared,
        'layer_mean': False,
        'dropout_masks': None,
        'alpha': alpha,
    }
    return model, batch, context


def _forward_with_trace(model, batch, context, monkeypatch):
    interests = []
    dropout_values = []

    def interest_hook(module, arguments):
        interests.append(arguments[0])

    def dropout_hook(module, arguments, result):
        dropout_values.append((arguments[0].detach().clone(), result.detach().clone()))

    handles = [
        model._sequential_projector.register_forward_pre_hook(interest_hook),
        model._dropout.register_forward_hook(dropout_hook),
    ]
    if not context['shared']:
        handles.append(model._graph_projector.register_forward_pre_hook(interest_hook))

    def deterministic_draws(count, **options):
        return torch.linspace(0.05, 0.95, count, device=options.get('device'), dtype=torch.float64)

    with monkeypatch.context() as patch, torch.random.fork_rng(devices=[]):
        patch.setattr(torch, 'rand', deterministic_draws)
        torch.manual_seed(812)
        outputs = model(batch)
    for handle in handles:
        handle.remove()
    outputs['raw_sequential'], outputs['raw_general'] = interests
    if context['regularized']:
        original, dropped = dropout_values[0]
        valid = torch.arange(original.shape[1])[None, :] < batch['item.length'][:, None]
        assert bool((original[valid].abs() > 1e-8).all())
        scale = (dropped != 0).double() / 0.7
        torch.testing.assert_close(dropped[valid], (original * scale)[valid], atol=1e-12, rtol=1e-12)
        context['dropout_masks'] = scale.numpy()
        adjusted = []
        for operator in context['operators']:
            locations = np.nonzero(operator)
            keep = np.linspace(0.05, 0.95, len(locations[0])) < 0.7
            weighted = operator.copy()
            weighted[locations] *= keep / 0.7
            adjusted.append(weighted)
        context['operators'] = tuple(adjusted)
    return outputs


def _architecture_oracle(parameters, context, positional_values=False, forward_positions=False):
    items = parameters['_item_embeddings.weight']
    users = parameters['_user_embeddings.weight']
    common, user_graph, item_graph = context['operators']
    propagated = _propagate(common, np.concatenate((users, items)), layer_mean=context['layer_mean'])
    propagated_users, propagated_items = propagated[: len(users)], propagated[len(users) :]
    sequential = []
    general = []
    for row, (user_id, history) in enumerate(zip(context['user_ids'], context['histories'])):
        history = np.asarray(history)
        positions = np.arange(len(history)) if forward_positions else np.arange(len(history) - 1, -1, -1)
        positioned = items[history] + parameters['_position_embeddings.weight'][positions]
        encoded = positioned.copy()
        if context['regularized']:
            centered = encoded - encoded.mean(axis=1, keepdims=True)
            encoded = centered / np.sqrt(np.mean(centered**2, axis=1, keepdims=True) + 1e-9)
            encoded = encoded * parameters['_layernorm.weight'] + parameters['_layernorm.bias']
            encoded *= context['dropout_masks'][row, : len(history)]
        hidden = np.tanh(encoded @ parameters['_current_interest_learning_encoder.0.weight'].T)
        attention = _softmax((hidden @ parameters['_current_interest_learning_encoder.2.weight'].T).ravel())
        sequential.append(attention @ (positioned if positional_values else items[history]))
        query = np.tanh(parameters['_general_interest_learning_encoder.0.weight'] @ propagated_users[user_id])
        graph_history = propagated_items[history]
        graph_attention = _softmax(graph_history @ query)
        general.append(graph_attention @ graph_history)
    sequential, general = np.asarray(sequential), np.asarray(general)
    unique_items = np.unique([item for history in context['histories'] for item in history])
    user_ids = list(context['user_ids'])
    graph_projector = '_sequential_projector' if context['shared'] else '_graph_projector'
    return {
        'raw_sequential': sequential,
        'raw_general': general,
        'combined_representation': context['alpha'] * sequential + (1 - context['alpha']) * general,
        'sequential_representation': _project(sequential, parameters, '_sequential_projector'),
        'graph_representation': _project(general, parameters, graph_projector),
        'common_graph_user_embeddings': _project(propagated_users[user_ids], parameters, '_user_projection'),
        'user_graph_user_embeddings': _project(
            _propagate(user_graph, users, layer_mean=context['layer_mean'])[user_ids], parameters, '_user_projection'
        ),
        'common_graph_item_embeddings': _project(propagated_items[unique_items], parameters, '_item_projection'),
        'item_graph_item_embeddings': _project(
            _propagate(item_graph, items, layer_mean=context['layer_mean'])[unique_items],
            parameters,
            '_item_projection',
        ),
    }


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('probe', ('readouts', 'projections', 'joint'))
def test_full_forward_and_all_parameter_gradients_match_numpy(profile, probe, monkeypatch):
    model, batch, context = _fixture(profile)
    actual = _forward_with_trace(model, batch, context, monkeypatch)
    parameters = {name: parameter.detach().numpy().copy() for name, parameter in model.named_parameters()}
    expected = _architecture_oracle(parameters, context)
    value_error = 0.0
    for name, values in expected.items():
        observed = actual[name].detach().numpy()
        np.testing.assert_allclose(observed, values, atol=2e-12, rtol=2e-12, err_msg=name)
        value_error = max(value_error, float(np.max(np.abs(observed - values))))
    objective = _torch_objective(actual, probe)
    assert objective.item() == pytest.approx(_objective(expected, probe), abs=2e-11)
    gradient_error, coordinates = _compare_gradients(
        objective, model.named_parameters(), lambda values: _objective(_architecture_oracle(values, context), probe)
    )
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    _emit(
        profile + '/' + probe,
        objective=objective.item(),
        max_value_error=value_error,
        max_gradient_error=gradient_error,
        parameter_coordinates=coordinates,
        unique_items=actual['item_ids'].numel(),
        user_ids=batch['user.ids'].tolist(),
    )


@pytest.mark.parametrize('layer_mean', (False, True))
@pytest.mark.parametrize('dropout', (0.0, 0.3))
def test_eq1_asymmetric_operator_feature_and_edge_gradients(layer_mean, dropout, monkeypatch):
    adjacency = np.array([[0, 5, 1, 0], [5, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0]], dtype=float)
    normalized = GraphDataset.get_sparse_graph_layer(sparse.csr_matrix(adjacency), 4, 4).toarray()
    np.testing.assert_allclose(normalized, _normalize_adjacency(adjacency), atol=1e-14)
    locations = np.array(np.nonzero(normalized))
    edges = torch.tensor(normalized[tuple(locations)], dtype=torch.float64, requires_grad=True)
    features = torch.tensor([[0.2, -0.7], [1.1, 0.4], [-0.8, 0.9], [0.3, 0.6]], dtype=torch.float64, requires_grad=True)
    graph = torch.sparse_coo_tensor(torch.tensor(locations), edges, (4, 4)).coalesce()
    draws = np.array([0.2, 0.85, 0.4, 0.35])
    monkeypatch.setattr(torch, 'rand', lambda count, **options: torch.tensor(draws, dtype=torch.float64))
    stub = SimpleNamespace(training=True, _graph_dropout=dropout, _num_graph_layers=2)
    actual = MCLSRModel._apply_graph_encoder(stub, features, graph, use_mean=layer_mean)

    def reference(values):
        operator = np.zeros((4, 4))
        operator[tuple(locations)] = values['edges'] * (draws < 1 - dropout) / (1 - dropout)
        return _propagate(operator, values['features'], layer_mean=layer_mean)

    expected = reference({'edges': edges.detach().numpy(), 'features': features.detach().numpy()})
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12)
    assert np.linalg.norm(expected) > 0.1
    objective = _torch_objective(
        {'combined_representation': actual, 'raw_sequential': actual, 'raw_general': actual}, 'readouts'
    )
    error, coordinates = _compare_gradients(
        objective, (('features', features), ('edges', edges)), lambda values: 6 * _probe(reference(values))
    )
    _emit(
        'eq1',
        layer_mean=layer_mean,
        dropout=dropout,
        output=actual.detach().tolist(),
        max_gradient_error=error,
        parameter_coordinates=coordinates,
    )


@pytest.mark.parametrize('alpha', (0.0, 0.23, 1.0))
def test_fusion_coefficients_and_upstream_gradients(alpha, monkeypatch):
    model, batch, context = _fixture(alpha=alpha)
    outputs = _forward_with_trace(model, batch, context, monkeypatch)
    fused = outputs['combined_representation']
    expected = alpha * outputs['raw_sequential'] + (1 - alpha) * outputs['raw_general']
    torch.testing.assert_close(fused, expected, atol=1e-12, rtol=1e-12)
    objective = _SyntheticPredictionProbe()(outputs)
    sequential_gradient, general_gradient = torch.autograd.grad(
        objective, (outputs['raw_sequential'], outputs['raw_general']), retain_graph=True
    )
    coefficients = torch.sin(torch.arange(fused.numel(), dtype=fused.dtype).reshape(fused.shape) + 0.4)
    derivative = (coefficients + 0.34 * fused) / fused.numel()
    torch.testing.assert_close(sequential_gradient, alpha * derivative, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(general_gradient, (1 - alpha) * derivative, atol=1e-12, rtol=1e-12)
    error, coordinates = _compare_gradients(
        objective,
        model.named_parameters(),
        lambda parameters: _probe(_architecture_oracle(parameters, context)['combined_representation']),
    )
    swapped = (1 - alpha) * outputs['raw_sequential'] + alpha * outputs['raw_general']
    swapped_delta = float((fused - swapped).abs().max().detach())
    assert swapped_delta > 1e-3
    _emit(
        'fusion_coefficients',
        alpha=alpha,
        swapped_delta=swapped_delta,
        max_gradient_error=error,
        parameter_coordinates=coordinates,
    )


def test_raw_value_readout_and_unprojected_fusion_reject_alternative_oracles(monkeypatch):
    model, batch, context = _fixture()
    actual = _forward_with_trace(model, batch, context, monkeypatch)
    parameters = {name: parameter.detach().numpy().copy() for name, parameter in model.named_parameters()}
    raw = _architecture_oracle(parameters, context)
    positional = _architecture_oracle(parameters, context, positional_values=True)
    forward_positions = _architecture_oracle(parameters, context, forward_positions=True)
    positional_delta = np.max(np.abs(raw['raw_sequential'] - positional['raw_sequential']))
    position_direction_delta = np.max(np.abs(raw['raw_sequential'] - forward_positions['raw_sequential']))
    projected_fusion = 0.5 * (actual['sequential_representation'] + actual['graph_representation'])
    fusion_delta = float((actual['combined_representation'] - projected_fusion).abs().max().detach())
    assert positional_delta > 1e-3
    assert position_direction_delta > 1e-3
    assert fusion_delta > 1e-3
    actual_gradient = torch.autograd.grad(actual['raw_sequential'].square().sum(), model._position_embeddings.weight)[0]
    assert actual_gradient.abs().max() > 1e-4
    _emit(
        'readout_counterfactuals',
        positional_value_delta=float(positional_delta),
        position_direction_delta=float(position_direction_delta),
        projected_fusion_delta=fusion_delta,
        position_gradient_max=float(actual_gradient.abs().max()),
    )


def test_shared_projector_accumulates_both_view_gradients(monkeypatch):
    model, batch, context = _fixture()
    assert model._sequential_projector is model._graph_projector
    outputs = _forward_with_trace(model, batch, context, monkeypatch)
    projector_parameters = tuple(model._sequential_projector.parameters())
    first = outputs['sequential_representation'].square().sum()
    second = outputs['graph_representation'].square().sum()
    combined_gradients = torch.autograd.grad(first + second, projector_parameters, retain_graph=True)
    first_gradients = torch.autograd.grad(first, projector_parameters, retain_graph=True)
    second_gradients = torch.autograd.grad(second, projector_parameters)
    for combined, sequential, general in zip(combined_gradients, first_gradients, second_gradients):
        torch.testing.assert_close(combined, sequential + general, atol=1e-12, rtol=1e-12)
    assert first_gradients[0].norm() > 1e-4 and second_gradients[0].norm() > 1e-4
    untied, _, _ = _fixture(shared=False)
    assert untied._sequential_projector is not untied._graph_projector
    _emit(
        'shared_projector',
        sequential_gradient_norm=float(first_gradients[0].norm()),
        graph_gradient_norm=float(second_gradients[0].norm()),
    )


@pytest.mark.parametrize('histories', (((1, 2, 1), (2,), (1, 2)), ((1, 2, 1), (2, 4), (3, 5, 2))))
def test_actual_item_domain_and_feature_sum_mean_scaling(histories, monkeypatch):
    model, batch, context = _fixture(histories=histories)
    outputs = _forward_with_trace(model, batch, context, monkeypatch)
    unique_items = sorted({item for history in histories for item in history})
    assert outputs['item_ids'].tolist() == unique_items
    user_mean, item_mean = [loss(outputs) for loss in _auxiliary_losses()[1:]]
    user_sum, item_sum = [loss(outputs) for loss in _auxiliary_losses(mean=False)[1:]]
    row_count = len(histories)
    torch.testing.assert_close(user_sum, row_count * user_mean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(item_sum, len(unique_items) * item_mean, atol=1e-12, rtol=1e-12)
    mean_gradient = torch.autograd.grad(item_mean, model._item_embeddings.weight, retain_graph=True)[0]
    sum_gradient = torch.autograd.grad(item_sum, model._item_embeddings.weight, retain_graph=True)[0]
    torch.testing.assert_close(sum_gradient, len(unique_items) * mean_gradient, atol=1e-12, rtol=1e-12)
    for loss, pair in zip((user_mean, item_mean), VIEW_PAIRS[1:]):
        assert loss.item() == pytest.approx(
            _paper_contrastive(*(outputs[name].detach().numpy() for name in pair)), abs=1e-12
        )
    actual_feature = 0.05 * (user_mean + item_mean)
    literal_batch_scaled = 0.05 * (user_sum + item_sum) / row_count
    assert abs(actual_feature.item() - literal_batch_scaled.item()) > 1e-3
    _emit(
        'feature_domain',
        unique_items=len(unique_items),
        user_rows=row_count,
        user_mean=user_mean.item(),
        item_mean=item_mean.item(),
        weighted_feature=actual_feature.item(),
        literal_sum_divided_by_rows=literal_batch_scaled.item(),
    )


def test_repeated_user_full_forward_differs_from_entity_masked_paper_oracle(monkeypatch):
    model, batch, context = _fixture(histories=((1, 2, 1), (2,), (1, 2)))
    batch['user.ids'] = torch.tensor([1, 1, 3])
    context['user_ids'] = (1, 1, 3)
    outputs = _forward_with_trace(model, batch, context, monkeypatch)
    torch.testing.assert_close(outputs['common_graph_user_embeddings'][0], outputs['common_graph_user_embeddings'][1])
    torch.testing.assert_close(outputs['user_graph_user_embeddings'][0], outputs['user_graph_user_embeddings'][1])
    observed = _auxiliary_losses()[1](outputs)
    parameters = {name: parameter.detach().numpy().copy() for name, parameter in model.named_parameters()}

    def entity_objective(values):
        reference = _architecture_oracle(values, context)
        return _paper_contrastive(
            reference[VIEW_PAIRS[1][0]], reference[VIEW_PAIRS[1][1]], entity_ids=context['user_ids']
        )

    entity_value = entity_objective(parameters)
    actual_gradient = torch.autograd.grad(observed, model._user_embeddings.weight)[0].numpy()
    entity_gradient = _finite_differences(entity_objective, parameters)['_user_embeddings.weight']
    gradient_delta = float(np.max(np.abs(actual_gradient - entity_gradient)))
    assert observed.item() > entity_value + 0.1
    assert gradient_delta > 1e-5
    _emit(
        'duplicate_user_domain',
        row_loss=observed.item(),
        entity_masked_loss=entity_value,
        user_embedding_gradient_delta=gradient_delta,
    )


def test_item_occurrence_multiplicity_does_not_reweight_unique_feature_loss(monkeypatch):
    histories = (((1, 2), (2, 1), (1,)), ((1, 1, 2), (2, 2, 1), (1, 1, 1)))
    results = []
    for batch_histories in histories:
        model, batch, context = _fixture(histories=batch_histories)
        outputs = _forward_with_trace(model, batch, context, monkeypatch)
        loss = _auxiliary_losses()[2](outputs)
        gradient = torch.autograd.grad(loss, model._item_embeddings.weight)[0]
        results.append((outputs['common_graph_item_embeddings'], loss, gradient))
    for original, repeated in zip(results[0], results[1]):
        torch.testing.assert_close(original, repeated, atol=1e-12, rtol=1e-12)
    _emit('item_multiplicity', events=[5, 9], unique_items=2, loss=results[0][1].item())


def test_graph_construction_characterizes_binary_counts_and_removed_diagonals(monkeypatch):
    raw_graphs = []
    normalize = GraphDataset.get_sparse_graph_layer

    def capture_graph(matrix, first_dim, second_dim, biparite=False):
        raw_graphs.append(matrix.toarray())
        return normalize(matrix, first_dim, second_dim, biparite=biparite)

    monkeypatch.setattr(GraphDataset, '_load_cached_graph', staticmethod(lambda *arguments: None))
    monkeypatch.setattr(GraphDataset, '_save_cached_graph', staticmethod(lambda *arguments: None))
    monkeypatch.setattr(GraphDataset, 'get_sparse_graph_layer', staticmethod(capture_graph))
    sequences = ((1, [1, 1, 2]), (2, [1, 2, 3]), (3, [2, 4, 4]))
    train = SimpleNamespace(dataset=[{'user.ids': [user], 'item.ids': items} for user, items in sequences])
    empty = SimpleNamespace(dataset=[])
    dataset = SimpleNamespace(num_users=3, num_items=4, meta={}, get_samplers=lambda: (train, empty, empty))
    built = GraphDataset(dataset, '/unused-architecture-fixture', use_user_graph=True, use_item_graph=True)
    counts = np.array([[2, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 2]], dtype=float)
    binary = (counts > 0).astype(float)
    user_coactions = binary @ binary.T
    item_coactions = binary.T @ binary
    np.fill_diagonal(user_coactions, 0)
    np.fill_diagonal(item_coactions, 0)
    np.testing.assert_array_equal(raw_graphs[0][1:4, 1:5], binary)
    np.testing.assert_array_equal(raw_graphs[1][1:4, 1:4], user_coactions)
    np.testing.assert_array_equal(raw_graphs[2][1:5, 1:5], item_coactions)
    assert (counts @ counts.T)[0, 1] == 3 and user_coactions[0, 1] == 2
    assert np.trace(binary @ binary.T) == 7 and np.trace(raw_graphs[1]) == 0
    common = np.block([[np.zeros((5, 5)), raw_graphs[0]], [raw_graphs[0].T, np.zeros((6, 6))]])
    for key, adjacency in zip(('graph', 'user_graph', 'item_graph'), (common, *raw_graphs[1:])):
        np.testing.assert_allclose(built.meta[key].to_dense().numpy(), _normalize_adjacency(adjacency), atol=1e-7)
    _emit(
        'binary_and_diagonal',
        raw_counts=counts.tolist(),
        stored_incidence=binary.tolist(),
        user_coactions=user_coactions.tolist(),
        item_coactions=item_coactions.tolist(),
        literal_count_coactions=(counts @ counts.T).tolist(),
    )


def test_top_k_row_pruning_direction_and_tie_policy_characterization():
    adjacency = np.array([[0, 5, 1], [5, 0, 3], [1, 3, 0]], dtype=float)
    retained = GraphDataset._filter_matrix_by_top_k(sparse.csr_matrix(adjacency), 1).toarray()
    expected = np.array([[0, 5, 0], [5, 0, 0], [0, 3, 0]], dtype=float)
    np.testing.assert_array_equal(retained, expected)
    normalized = GraphDataset.get_sparse_graph_layer(sparse.csr_matrix(retained), 3, 3).toarray()
    np.testing.assert_allclose(normalized, _normalize_adjacency(expected), atol=1e-14)
    features = np.array([[1.0], [2.0], [4.0]])
    received = normalized @ features
    assert received[2, 0] == pytest.approx(2 * np.sqrt(3 / 5))
    assert not np.allclose(received, normalized.T @ features)
    ties = sparse.csr_matrix([[0, 2, 2], [2, 0, 2], [2, 2, 0]])
    tied_result = GraphDataset._filter_matrix_by_top_k(ties, 1).toarray()
    np.testing.assert_array_equal(np.count_nonzero(tied_result, axis=1), [1, 1, 1])
    assert np.all(tied_result.sum(axis=1) == 2)
    _emit(
        'top_k_direction',
        retained=retained.tolist(),
        propagated=received.ravel().tolist(),
        transposed_propagated=(normalized.T @ features).ravel().tolist(),
        tied_result=tied_result.tolist(),
    )
