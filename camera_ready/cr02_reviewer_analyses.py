"""Camera-ready (BIBM 2026): additional analyses requested by reviewers.

  A. Graph-construction robustness of the modularity gap and of Delta-Q (R1.3, R4.2)
  B. Continuous-score ridge regression on the 90 neural features (R3.1)
  C. Permutation-null mean/SD for the five classifiers (R3.3)
  D. Sensitivity to the one subject with missing 2-back behaviour (R1.4)

Inputs : camera_ready/fc30.npz (cr01), results/latest/efficiency/neural_efficiency.csv
Outputs: camera_ready/cr02_results.json and a printed summary.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)
from scripts.s03_connectivity_analysis import (  # noqa: E402
    get_network_parcels, compute_modularity, compute_global_efficiency)

OUT = ROOT / "camera_ready"
SEED = 42
res = {}


def welch(h, l):
    t, p = stats.ttest_ind(h, l, equal_var=False)
    sp = np.sqrt(((len(h) - 1) * np.var(h, ddof=1) + (len(l) - 1) * np.var(l, ddof=1)) / (len(h) + len(l) - 2))
    return {"t": float(t), "p": float(p), "d": float((np.mean(h) - np.mean(l)) / sp)}


df = pd.read_csv(ROOT / "results/latest/efficiency/neural_efficiency.csv")
df["subject"] = df["subject"].astype(str)
high = (df["efficiency_group"] == "High_Efficiency").values

# ---------------------------------------------------------------- A. robustness
fc = np.load(OUT / "fc30.npz")
names = list(fc["parcel_names"])
nets = get_network_parcels()
apriori = [i for i, (net, parcels) in enumerate(nets.items()) for _ in parcels]
assert [p for ps in nets.values() for p in ps] == names


def binary_abs(z, thr):
    a = (np.abs(z) > thr).astype(float)
    np.fill_diagonal(a, 0)
    return a


def binary_pos(z, thr):
    a = (z > thr).astype(float)
    np.fill_diagonal(a, 0)
    return a


def proportional(z, dens):
    iu = np.triu_indices_from(z, 1)
    k = int(round(dens * len(iu[0])))
    cut = np.sort(z[iu])[::-1][k - 1]
    a = (z >= cut).astype(float)
    np.fill_diagonal(a, 0)
    return a


def weighted_pos(z):
    a = np.clip(z, 0, None)
    np.fill_diagonal(a, 0)
    return a


def q_apriori(a):
    return compute_modularity(a, apriori)


def q_louvain(a, n_seeds=20):
    g = nx.from_numpy_array(a)
    if g.number_of_edges() == 0:
        return 0.0
    qs = [nx.community.modularity(g, nx.community.louvain_communities(g, weight="weight", seed=s), weight="weight")
          for s in range(n_seeds)]
    return float(np.mean(qs))


VARIANTS = {
    "abs|z|>0.15 binary, a priori (original)": (lambda z: binary_abs(z, 0.15), q_apriori),
    "abs|z|>0.05 binary, a priori": (lambda z: binary_abs(z, 0.05), q_apriori),
    "abs|z|>0.10 binary, a priori": (lambda z: binary_abs(z, 0.10), q_apriori),
    "abs|z|>0.20 binary, a priori": (lambda z: binary_abs(z, 0.20), q_apriori),
    "abs|z|>0.25 binary, a priori": (lambda z: binary_abs(z, 0.25), q_apriori),
    "z>0.15 positive-only binary, a priori": (lambda z: binary_pos(z, 0.15), q_apriori),
    "density 20% binary, a priori": (lambda z: proportional(z, 0.20), q_apriori),
    "density 30% binary, a priori": (lambda z: proportional(z, 0.30), q_apriori),
    "density 40% binary, a priori": (lambda z: proportional(z, 0.40), q_apriori),
    "weighted positive, a priori": (weighted_pos, q_apriori),
    "abs|z|>0.15 binary, Louvain (20 seeds)": (lambda z: binary_abs(z, 0.15), q_louvain),
    "weighted positive, Louvain (20 seeds)": (weighted_pos, q_louvain),
}

subs = df["subject"].tolist()
rob = {}
for vname, (build, qf) in VARIANTS.items():
    # metric per run, then averaged over available runs (as in s03)
    Q = {c: np.array([np.mean([qf(build(fc[f"{s}|{c}|{r}"])) for r in ("LR", "RL") if f"{s}|{c}|{r}" in fc])
                      for s in subs]) for c in ["full", "0bk", "2bk"]}
    dQ = Q["2bk"] - Q["0bk"]
    rob[vname] = {
        "Q_run": welch(Q["full"][high], Q["full"][~high]),
        "Q_0bk": welch(Q["0bk"][high], Q["0bk"][~high]),
        "Q_2bk": welch(Q["2bk"][high], Q["2bk"][~high]),
        "Q_condavg": welch(((Q["0bk"] + Q["2bk"]) / 2)[high], ((Q["0bk"] + Q["2bk"]) / 2)[~high]),
        "dQ": welch(dQ[high], dQ[~high]),
        "dQ_mean_high": float(dQ[high].mean()), "dQ_mean_low": float(dQ[~high].mean()),
        "dQ_onesample_all_p": float(stats.ttest_1samp(dQ, 0).pvalue),
    }
    if "original" in vname:  # must reproduce the manuscript's t=2.19 from neural_efficiency.csv
        assert np.allclose(Q["full"], df["modularity"].values, atol=1e-8), "reproduction failed"
        rob[vname]["reproduces_csv"] = True
        # D. sensitivity: drop subject with missing 2-back behaviour
        keep = df["acc_2bk"].notna().values
        res["drop_missing_subject_Q_run"] = welch(Q["full"][high & keep], Q["full"][~high & keep])
        # permutation p for whole-run Q
        rng = np.random.default_rng(SEED)
        obs = Q["full"][high].mean() - Q["full"][~high].mean()
        null = [(lambda m: Q["full"][m].mean() - Q["full"][~m].mean())(rng.permutation(high)) for _ in range(10000)]
        rob[vname]["Q_run_perm_p"] = float((np.sum(np.abs(null) >= abs(obs)) + 1) / 10001)
    print(f"{vname:45s} Qrun t={rob[vname]['Q_run']['t']:+.2f} p={rob[vname]['Q_run']['p']:.3f} "
          f"d={rob[vname]['Q_run']['d']:+.2f} | Qavg d={rob[vname]['Q_condavg']['d']:+.2f} "
          f"p={rob[vname]['Q_condavg']['p']:.3f} | dQ p={rob[vname]['dQ']['p']:.2f}", flush=True)
res["robustness"] = rob

# ---------------------------------------------------- B/C. ML on 90 neural features
import scripts.s10_ai_analysis as s10  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import (KFold, StratifiedKFold, RepeatedKFold,  # noqa: E402
                                     cross_val_score, cross_val_predict, permutation_test_score)

dfa = s10.load_data_for_ai()
X, y, fnames = s10.prepare_features(dfa, feature_type="neural_only")
assert X.shape[1] == 90, X.shape
Xs = StandardScaler().fit_transform(X)

# C. permutation null for classifiers (identical settings to s10)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
clfs = {
    "Logistic Regression": s10.LogisticRegression(random_state=SEED, max_iter=1000, C=0.1),
    "SVM (RBF)": s10.SVC(kernel="rbf", probability=True, random_state=SEED, C=1.0),
    "Random Forest": s10.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED),
    "Gradient Boosting": s10.GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=SEED),
    "MLP (64-32-16)": s10.MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=SEED,
                                        early_stopping=True, validation_fraction=0.2),
}
perm = {}
for n, c in clfs.items():
    sc, ps, p = permutation_test_score(c, Xs, y, cv=cv, n_permutations=200, scoring="accuracy",
                                       random_state=SEED, n_jobs=-1)
    perm[n] = {"acc": float(sc), "p": float(p), "null_mean": float(np.mean(ps)), "null_sd": float(np.std(ps)),
               "z_vs_null": float((sc - np.mean(ps)) / np.std(ps))}
    print(f"{n:22s} acc={sc:.3f} p={p:.3f} null={np.mean(ps):.3f}+/-{np.std(ps):.3f}", flush=True)
res["permutation_null"] = perm

# B. continuous regression (nested: RidgeCV picks alpha inside each training fold)
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 40)))
reg = {}
for target in ["composite_efficiency", "ies_2bk"]:
    yt = dfa[target].values
    ok = ~np.isnan(yt)
    Xt, yt = X[ok], yt[ok]
    r2 = cross_val_score(ridge, Xt, yt, cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED), scoring="r2")
    kf = KFold(5, shuffle=True, random_state=SEED)
    sc, ps, p = permutation_test_score(ridge, Xt, yt, cv=kf, scoring="r2", n_permutations=1000,
                                       random_state=SEED, n_jobs=-1)
    pred = cross_val_predict(ridge, Xt, yt, cv=kf)
    r, rp = stats.pearsonr(pred, yt)
    reg[target] = {"n": int(ok.sum()), "cv_r2_mean": float(r2.mean()), "cv_r2_sd": float(r2.std()),
                   "r2_single_5fold": float(sc), "perm_p": float(p), "null_r2_mean": float(np.mean(ps)),
                   "pred_obs_r": float(r), "pred_obs_p": float(rp)}
    print(f"ridge -> {target}: R2={r2.mean():.3f}+/-{r2.std():.3f} perm p={p:.3f} r(pred,obs)={r:.3f}", flush=True)
res["ridge_regression"] = reg

# label bookkeeping (R1.4)
res["labels"] = {
    "split_variable": "composite_efficiency = mean z(ACC_2bk, -RT_2bk, -ACC_cost, -RT_cost), pooled over LR+RL runs",
    "n_high": int(high.sum()), "n_low": int((~high).sum()),
    "agreement_with_IES2bk_median_split": float(
        ((df["ies_2bk"] <= df["ies_2bk"].median()) == high)[df["ies_2bk"].notna()].mean()),
    "missing_2bk_subject": df.loc[df["acc_2bk"].isna(), ["subject", "efficiency_group"]].to_dict("records"),
}
json.dump(res, open(OUT / "cr02_results.json", "w"), indent=2)
print(json.dumps(res["labels"], indent=2))
print("drop-missing:", res["drop_missing_subject_Q_run"])
