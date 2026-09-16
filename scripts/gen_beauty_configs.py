import json
import os

SRC = 'configs/train/toys'
DST = 'configs/train/beauty'
os.makedirs(DST, exist_ok=True)


def patch_metrics(md):
    # закрыть дыру toys: у sasrec-армов не логировались @100/@1000
    for k in (100, 1000):
        md['ndcg@%d' % k] = {'type': 'mclsr-ndcg', 'k': k}
        md['recall@%d' % k] = {'type': 'mclsr-recall', 'k': k}
        md['hit@%d' % k] = {'type': 'mclsr-hit', 'k': k}


def rename_dataset(d):
    if isinstance(d, dict):
        if d.get('name') == 'Toys':
            d['name'] = 'Beauty'
        for v in d.values():
            rename_dataset(v)


for fn in sorted(os.listdir(SRC)):
    if not fn.endswith('.json'):
        continue
    s = open(os.path.join(SRC, fn)).read().replace('data/Toys', 'data/Beauty')
    c = json.loads(s)
    name = c['experiment_name'].replace('toys_', 'beauty_').replace('_Toys', '_Beauty')
    if not name.endswith('_Beauty'):
        name += '_Beauty'
    c['experiment_name'] = name
    rename_dataset(c)
    for cb in c['callback']['callbacks']:
        if fn.startswith('sasrec') and cb.get('type') in ('validation', 'eval'):
            patch_metrics(cb['metrics'])
        if cb.get('type') == 'eval':
            # test on the validation grid: "test @ val-best" becomes exact
            # (toys logged test every 256 steps -> nearest-event approximation)
            cb['on_step'] = 64
    with open(os.path.join(DST, fn), 'w') as f:
        json.dump(c, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print('wrote', fn, '->', name)
