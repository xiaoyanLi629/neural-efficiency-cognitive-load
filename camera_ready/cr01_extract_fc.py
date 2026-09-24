"""Camera-ready (BIBM 2026): re-extract and SAVE parcel-level FC matrices.

s03 computes graph metrics on the 30 Schaefer network parcels but never saves the
30x30 matrices, so graph-construction robustness checks (reviewers #1, #4) need them.
Uses exactly the s03 preprocessing (detrend + z-score, block extraction, Fisher z).
Output: camera_ready/fc30.npz with keys '<subject>|<full|0bk|2bk>|<LR|RL>' (per-run 30x30);
s03 computes graph metrics per run and then averages the metrics, so runs are kept separate.
"""
import sys
import logging
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)
from configs.config import get_wm_fmri_path, WM_TASK  # noqa: E402
from scripts.s03_connectivity_analysis import (  # noqa: E402
    get_network_parcels, extract_parcel_timeseries, preprocess_timeseries,
    extract_task_blocks, compute_connectivity_matrix)
from scripts.s02_activation_analysis import get_task_events  # noqa: E402

PARCELS = {n: idx for net in get_network_parcels().values() for n, idx in net.items()}


def one(subject):
    out = {}
    for run in ["LR", "RL"]:
        p = get_wm_fmri_path(subject, run)
        if not p.exists():
            continue
        try:
            data = nib.load(str(p)).get_fdata()
            ev = get_task_events(subject, run)
            ts = {n: preprocess_timeseries(extract_parcel_timeseries(data, v))
                  for n, v in PARCELS.items()}
            out[f"full|{run}"] = compute_connectivity_matrix(ts)[0]
            for c in ["0bk", "2bk"]:
                tc = {n: extract_task_blocks(t, ev, c, WM_TASK["tr"]) for n, t in ts.items()}
                if all(len(t) > 10 for t in tc.values()):
                    out[f"{c}|{run}"] = compute_connectivity_matrix(tc)[0]
        except Exception as e:
            print(f"ERR {subject} {run}: {e}", flush=True)
    return subject, out


if __name__ == "__main__":
    import pandas as pd
    subs = pd.read_csv(ROOT / "results/latest/efficiency/neural_efficiency.csv")["subject"].astype(str).tolist()
    res = {}
    with Pool(24) as pool:
        for i, (s, m) in enumerate(pool.imap_unordered(one, subs), 1):
            for k, v in m.items():
                res[f"{s}|{k}"] = v
            print(f"{i}/{len(subs)} {s} {len(m)}", flush=True)
    np.savez_compressed(ROOT / "camera_ready/fc30.npz",
                        parcel_names=np.array(list(PARCELS)), **res)
    print("DONE", len(res))
