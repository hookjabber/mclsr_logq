import pytest
import torch

from irec.models.sasrec import SASRecModel


def _model(num_items):
    return SASRecModel(
        sequence_prefix='item',
        positive_prefix='labels',
        num_items=num_items,
        max_sequence_length=8,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    ).train()


@pytest.mark.parametrize('accept_final_draw', [False, True])
def test_rejection_budget_does_not_reject_an_eligible_catalogue(monkeypatch, accept_final_draw):
    model = _model(3)
    calls = []

    def controlled_randint(low, high, *, size, device):
        calls.append((low, high))
        if low == 0:
            value = 0
        elif accept_final_draw and len(calls) == 65:
            value = 3
        else:
            value = 1
        return torch.full(size, value, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, 'randint', controlled_randint)
    output = model(
        {
            'item.ids': torch.tensor([1]),
            'item.length': torch.tensor([1]),
            'labels.ids': torch.tensor([2]),
        }
    )
    assert output['negative_ids'].tolist() == [3]
    assert len(calls) == (65 if accept_final_draw else 66)
    (output['positive_scores'].sum() + output['negative_scores'].sum()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


@pytest.mark.parametrize('fallback_index', [0, -1])
def test_exact_fallback_respects_each_users_full_sequence(monkeypatch, fallback_index):
    model = _model(5)

    def controlled_randint(low, high, *, size, device):
        value = 2 if low == 1 else (0 if fallback_index == 0 else high - 1)
        return torch.full(size, value, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, 'randint', controlled_randint)
    output = model(
        {
            'item.ids': torch.tensor([1, 1, 2, 2]),
            'item.length': torch.tensor([2, 2]),
            'labels.ids': torch.tensor([1, 2, 2, 3]),
        }
    )
    expected = [3, 3, 1, 1] if fallback_index == 0 else [5, 5, 5, 5]
    assert output['negative_ids'].tolist() == expected


def test_fallback_rejects_genuinely_saturated_user():
    model = _model(3)
    with pytest.raises(RuntimeError, match='no eligible items'):
        model(
            {
                'item.ids': torch.tensor([1, 1, 2]),
                'item.length': torch.tensor([1, 2]),
                'labels.ids': torch.tensor([2, 2, 3]),
            }
        )
