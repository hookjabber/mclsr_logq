"""Print validation and test ndcg@20 along training for TensorBoard runs matching the given patterns.

    python scripts/val_test_curves.py "beauty_sasrec_inbatch_logq_Beauty_2026-09-13*"
"""
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for pat in sys.argv[1:]:
    for d in sorted(glob.glob("tensorboard_logs/" + pat)):
        acc = EventAccumulator(d, size_guidance={"scalars": 0}); acc.Reload()
        tags = acc.Tags().get("scalars", [])
        vt = [t for t in tags if t.startswith("validation/ndcg@20")]; et = [t for t in tags if t.startswith("eval/ndcg@20")]
        if not vt: print(d, "tags:", tags[:8]); continue
        v = {e.step: e.value for e in acc.Scalars(vt[0])}; e = {x.step: x.value for x in acc.Scalars(et[0])} if et else {}
        print("==", d.split("/")[-1])
        for s in sorted(v):
            if s % 320 == 0 or s in (704, 1344, 2432, 2496):
                print("  step %5d  val %.4f  test %s" % (s, v[s], ("%.4f" % e[s]) if s in e else "-"))
