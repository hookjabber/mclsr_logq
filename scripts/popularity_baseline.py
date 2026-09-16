"""MostPop anchor on the MCLSR test protocol plus an independent numpy re-computation of the metrics.

    python scripts/popularity_baseline.py configs/train/beauty/01_orig.json
"""
import json, os, sys, pickle, numpy as np, torch
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from irec.dataloader import BaseDataloader
from irec.dataset import BaseDataset
from irec.utils import DEVICE
from eval_checkpoint import evaluate, get_inference_config

params = sys.argv[1] if len(sys.argv) > 1 else 'configs/train/beauty/01_orig.json'
config = json.load(open(params))
dataset = BaseDataset.create_from_config(config['dataset'])
_, _, test_sampler = dataset.get_samplers()
loader = BaseDataloader.create_from_config(config['dataloader']['validation'], dataset=test_sampler, **dataset.meta)
counts = pickle.load(open(config['loss']['losses'][0]['path_to_item_counts'], 'rb'))
counts = np.asarray(counts['counts'] if isinstance(counts, dict) else counts, dtype=np.float64)
scores = torch.tensor(counts, dtype=torch.float32)
scores[0] = -1e12; scores[-1] = -1e12               # pad / mask columns, as in the model
K = config['model'].get('eval_top_k', 50)

class PopModel(torch.nn.Module):
    def forward(self, batch):
        n = batch['labels.length'].shape[0]
        return torch.topk(scores.to(batch['labels.length'].device), k=K).indices.unsqueeze(0).expand(n, K).clone()

inf = get_inference_config(config, 'eval')
res = evaluate(model=PopModel(), dataloader=loader, metric_configs=inf['metrics'], pred_prefix=inf['pred_prefix'], labels_prefix=inf['labels_prefix'], meta=dataset.meta)
print('MostPop via irec metrics:', {k: round(v, 4) for k, v in res.items() if k in ('ndcg@20', 'recall@1000', 'ndcg@50', 'recall@20', 'hit@20')})

# independent numpy recomputation on the same predictions
top = torch.topk(scores, k=K).indices.numpy()
nd, rc, n_users = [], [], 0
for batch in loader:
    lab, ln = batch['labels.ids'].numpy(), batch['labels.length'].numpy(); off = 0
    for L in ln:
        labels = set(lab[off:off + L].tolist()); off += L
        hits20 = [1.0 if t in labels else 0.0 for t in top[:20]]
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits20)); idcg = sum(1 / np.log2(i + 2) for i in range(min(20, L)))
        nd.append(dcg / idcg if idcg > 0 else 0.0)
        rc.append(sum(1 for t in top[:1000] if t in labels) / L if L > 0 else 0.0); n_users += 1
print('MostPop via numpy:        ndcg@20 %.4f  recall@1000 %.4f  (users %d)' % (np.mean(nd), np.mean(rc), n_users))
print('random-chance recall@1000 ~ %.4f' % (1000 / (len(counts) - 2)))
