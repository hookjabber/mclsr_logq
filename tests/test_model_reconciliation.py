import pytest
import torch

import irec.models.base as base_module
import irec.utils as utility_module
from irec.models import BaseModel


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(utility_module, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(base_module, 'DEVICE', torch.device('cpu'))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(174)
        yield


def _model(kind, **overrides):
    config = {
        'type': kind,
        'sequence_prefix': 'item',
        'user_prefix': 'user',
        'labels_prefix': 'labels',
        'positive_prefix': 'labels',
        'use_graph': False,
        'use_layernorm': False,
        'embedding_dim': 8,
        'num_heads': 2,
        'num_layers': 2,
        'num_graph_layers': 0,
        'dropout': 0.0,
        'eval_top_k': 6,
        **overrides,
    }
    return BaseModel.create_from_config(config, num_items=6, num_users=3, max_sequence_length=4)


def _batch(kind, histories):
    events = [item for history in histories for item in history]
    labels = [5] * len(histories) if kind == 'mclsr' else [item % 6 + 1 for item in events]
    return {
        'item.ids': torch.tensor(events, dtype=torch.long),
        'item.length': torch.tensor([len(history) for history in histories], dtype=torch.long),
        'labels.ids': torch.tensor(labels, dtype=torch.long),
        'user.ids': torch.arange(1, len(histories) + 1),
        'negatives.ids': torch.empty(0, dtype=torch.long),
    }


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'sasrec_in_batch'])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('histories', [[], [[]], [[1, 2], []]])
def test_empty_histories_fail_before_attention(kind, training, histories):
    model = _model(kind).train(training)
    with pytest.raises(ValueError, match='non-empty.*positive.*lengths'):
        model(_batch(kind, histories))


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec', 'sasrec_in_batch'])
@pytest.mark.parametrize('training', [False, True])
def test_packed_event_count_cannot_broadcast_across_histories(kind, training):
    model = _model(kind).train(training)
    batch = _batch(kind, [[1, 2], [3]])
    batch['item.ids'] = torch.tensor([1])
    with pytest.raises(ValueError, match='lengths.*number of events'):
        model(batch)


@pytest.mark.parametrize('use_layernorm', [False, True])
def test_current_interest_matches_positioned_keys_and_raw_values(use_layernorm):
    model = _model('mclsr', use_layernorm=use_layernorm).train()
    histories = [[1, 2, 1], [3], [2, 4]]
    batch = _batch('mclsr', histories)
    output = model(batch)
    expected = []
    for history in histories:
        raw = model._item_embeddings(torch.tensor(history))
        positions = torch.arange(len(history) - 1, -1, -1)
        positioned = model._layernorm(raw + model._position_embeddings(positions))
        scores = model._current_interest_learning_encoder(positioned).squeeze(-1)
        expected.append(torch.softmax(scores, dim=0) @ raw)
    torch.testing.assert_close(output['combined_representation'], torch.stack(expected))
    torch.testing.assert_close(output['combined_representation'][1], model._item_embeddings.weight[3])
    output['combined_representation'].square().sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert model._position_embeddings.weight.grad[2].abs().sum() > 0
    assert model._item_embeddings.weight.grad[0].count_nonzero() == 0


@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
def test_graph_off_keeps_unscaled_current_interest(alpha, monkeypatch):
    model = _model('mclsr', alpha=alpha).train()

    def forbidden_graph(*args, **kwargs):
        pytest.fail('graph-off forward invoked graph propagation')

    monkeypatch.setattr(model, '_apply_graph_encoder', forbidden_graph)
    output = model(_batch('mclsr', [[3], [1]]))
    torch.testing.assert_close(output['combined_representation'], model._item_embeddings.weight[[3, 1]])
    assert 'graph_representation' not in output
    assert 'sequential_representation' not in output


@pytest.mark.parametrize('share_il_projector', [False, True])
def test_inference_uses_current_interest_raw_dot_without_graph_or_projection(share_il_projector, monkeypatch):
    model = _model('mclsr', use_graph=True, alpha=0.0, share_il_projector=share_il_projector).eval()
    assert (model._graph_projector is model._sequential_projector) == share_il_projector

    def forbidden_branch(*args, **kwargs):
        pytest.fail('inference invoked a training-only graph/projector branch')

    monkeypatch.setattr(model, '_apply_graph_encoder', forbidden_branch)
    monkeypatch.setattr(model._sequential_projector, 'forward', forbidden_branch)
    monkeypatch.setattr(model._graph_projector, 'forward', forbidden_branch)
    with torch.no_grad():
        model._item_embeddings.weight.zero_()
        model._item_embeddings.weight[1, :2] = torch.tensor([1.0, 0.0])
        model._item_embeddings.weight[2, :2] = torch.tensor([2.0, 3.0])
        model._item_embeddings.weight[3, :2] = torch.tensor([0.5, 0.0])
        model._item_embeddings.weight[4, :2] = torch.tensor([-1.0, 0.0])
        model._item_embeddings.weight[5, :2] = torch.tensor([0.2, 0.0])
        model._item_embeddings.weight[6, :2] = torch.tensor([-2.0, 0.0])
        result = model(_batch('mclsr', [[1]]))
    assert result.tolist() == [[2, 1, 3, 5, 4, 6]]


def test_sasrec_last_queries_select_matching_all_position_targets_and_owners():
    all_model = _model('sasrec_in_batch', query_positions='all').train()
    last_model = _model('sasrec_in_batch', query_positions='last').train()
    last_model.load_state_dict(all_model.state_dict())
    batch = _batch('sasrec_in_batch', [[1, 2, 3], [4], [2, 5]])
    batch['user.ids'] = torch.tensor([2, 2, 3])
    all_output = all_model(batch)
    last_output = last_model(batch)
    last_indices = torch.tensor([2, 3, 5])
    torch.testing.assert_close(last_output['query_embeddings'], all_output['query_embeddings'][last_indices])
    torch.testing.assert_close(last_output['positive_embeddings'], all_output['positive_embeddings'][last_indices])
    torch.testing.assert_close(last_output['positive_ids'], batch['labels.ids'][last_indices])
    assert all_output['user_ids'].tolist() == [2, 2, 2, 2, 3, 3]
    assert last_output['user_ids'].tolist() == [2, 2, 3]
    assert isinstance(all_model._encoder, torch.nn.TransformerEncoder)


@pytest.mark.parametrize('kind', ['sasrec', 'sasrec_in_batch'])
def test_sasrec_causal_prefix_does_not_use_future_item_content(kind):
    model = _model(kind).train()
    original = torch.tensor([1, 2, 3, 4])
    changed = torch.tensor([1, 2, 5, 6])
    lengths = torch.tensor([4])
    original_hidden, _ = model._apply_sequential_encoder(original, lengths)
    changed_hidden, _ = model._apply_sequential_encoder(changed, lengths)
    torch.testing.assert_close(original_hidden[:, :2], changed_hidden[:, :2])
    assert not torch.allclose(original_hidden[:, 2:], changed_hidden[:, 2:])


@pytest.mark.parametrize('kind', ['mclsr', 'sasrec_in_batch'])
def test_right_padding_does_not_change_valid_queries_or_gradients(kind):
    model = _model(kind, query_positions='last').train()
    output_key = 'combined_representation' if kind == 'mclsr' else 'query_embeddings'
    batched = model(_batch(kind, [[1, 2, 3], [4]]))[output_key][1]
    standalone = model(_batch(kind, [[4]]))[output_key][0]
    torch.testing.assert_close(batched, standalone)
    parameters = tuple(model.parameters())
    batched_gradients = torch.autograd.grad(batched.square().sum(), parameters, allow_unused=True)
    standalone_gradients = torch.autograd.grad(standalone.square().sum(), parameters, allow_unused=True)
    for batched_gradient, standalone_gradient in zip(batched_gradients, standalone_gradients):
        if batched_gradient is None:
            assert standalone_gradient is None
        else:
            assert torch.isfinite(batched_gradient).all()
            torch.testing.assert_close(batched_gradient, standalone_gradient, atol=1e-6, rtol=1e-4)


def test_classic_sasrec_scores_every_valid_position_against_its_target():
    model = _model('sasrec').train()
    batch = _batch('sasrec', [[1, 2], [3]])
    hidden, mask = model._apply_sequential_encoder(batch['item.ids'], batch['item.length'])
    expected = (hidden[mask] * model._item_embeddings(batch['labels.ids'])).sum(dim=1)
    output = model(batch)
    torch.testing.assert_close(output['positive_scores'], expected)
    assert output['negative_scores'].shape == expected.shape
