"""Temporal validation of the frozen race-free PreventScreen model on NHANES August 2021-August 2023.

Usage:
    python temporal.py DIR
where DIR contains the official NCHS public files DEMO_L.xpt, GHB_L.xpt, BMX_L.xpt, BPXO_L.xpt,
HDL_L.xpt, TCHOL_L.xpt, DIQ_L.xpt, MCQ_L.xpt and PAQ_L.xpt, downloaded from
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/

The model in out/model_race_free.json is applied unchanged (no refitting). Family history of diabetes
(MCQ300C) is not released in this cycle, so it is set to the development median (0) for everyone.
Writes out/temporal_2021_2023.json.
"""
import json, sys, os
import numpy as np
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "."
M = json.load(open(os.path.join(os.path.dirname(__file__), "out", "model_race_free.json")))
MED, MEAN, SC, CO, IC, THR = (np.array(M[k], float) if isinstance(M[k], list) else M[k]
                              for k in ["medians", "mean", "scale", "coef", "intercept", "threshold_sens70"])

def rd(f):
    return pd.read_sas(os.path.join(D, f + ".xpt"), format="xport").set_index("SEQN")

demo = rd("DEMO_L")
df = demo[(demo.RIDAGEYR.between(18, 35)) & (demo.WTMEC2YR > 0)].copy()
n_age = len(df)
for f in ["GHB_L", "BMX_L", "BPXO_L", "HDL_L", "TCHOL_L", "DIQ_L", "PAQ_L"]:
    t = rd(f)
    df = df.join(t[[c for c in t.columns if c not in df.columns and c != "WTPH2YR"]], how="left")

dx = (df.DIQ010 == 1) | (df.DIQ050 == 1) | (df.DIQ070 == 1)
n_dx = int(dx.sum()); df = df[~dx]
n_noa1c = int(df.LBXGH.isna().sum()); df = df[df.LBXGH.notna()].copy()

clean = lambda s: s.where(s < 7777)
sbp = df[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].apply(clean).mean(axis=1)
dbp = df[["BPXODI1", "BPXODI2", "BPXODI3"]].apply(clean).mean(axis=1)
mod, vig = clean(df.PAD790Q), clean(df.PAD810Q)
active = np.where(mod.isna() & vig.isna(), np.nan, ((mod.fillna(0) > 0) | (vig.fillna(0) > 0)).astype(float))
X = np.column_stack([df.RIDAGEYR, (df.RIAGENDR == 1).astype(float), df.BMXBMI, df.BMXWAIST, sbp, dbp,
                     df.LBDHDD, df.LBXTC - df.LBDHDD, np.full(len(df), np.nan), active]).astype(float)
X = np.where(np.isnan(X), MED, X)
p = 1 / (1 + np.exp(-(IC + ((X - MEAN) / SC) @ CO)))

y = (df.LBXGH >= 5.7).astype(int).values
w = df.WTMEC2YR.values; st = df.SDMVSTRA.values; psu = df.SDMVPSU.values
asian = (df.RIDRETH3 == 6).values; b = df.BMXBMI.values
c1, c2, c3 = np.where(asian, 23, 25), np.where(asian, 27.5, 30), np.where(asian, 37.5, 40)
bmipts = np.select([b < c1, b < c2, b < c3], [0, 1, 2], 3)
htn = ((sbp.fillna(0) >= 140) | (dbp.fillna(0) >= 90)).astype(int).values
ada = np.where(np.isnan(b), np.nan, (df.RIAGENDR == 1).astype(int).values + htn + (active == 0).astype(int) + bmipts)
us = ((df.RIDAGEYR >= 35) & (df.BMXBMI >= 25)).astype(int).values
aware = ((df.DIQ160 == 1) | (df.DIQ010 == 3)).astype(int).values

T = {7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.16, 14: 2.145, 15: 2.131, 16: 2.12}
def ratio(v, dom=None):
    dom = np.ones(len(v), bool) if dom is None else dom
    v = np.nan_to_num(v); den = w[dom].sum(); R = (w * v)[dom].sum() / den
    z = np.where(dom, w * (v - R) / den, 0.0)
    g = pd.DataFrame({"s": st, "p": psu, "z": z}).groupby(["s", "p"]).z.sum().reset_index()
    var, dfree = 0.0, 0
    for _, a in g.groupby("s"):
        n = len(a)
        if n < 2: continue
        var += n / (n - 1) * ((a.z - a.z.mean()) ** 2).sum(); dfree += n - 1
    se = np.sqrt(var); t = 1.96 if dfree >= 30 else T.get(dfree, 2.0)
    return [R, R - t * se, R + t * se]

def auc(s, yy, ww):
    m = ~np.isnan(s); s, yy, ww = s[m], yy[m], ww[m]
    o = np.argsort(s, kind="mergesort"); s, yy, ww = s[o], yy[o], ww[o]
    P, N = (ww * yy).sum(), (ww * (1 - yy)).sum(); cumN = tot = 0.0; i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]: j += 1
        tp, tn = (ww[i:j] * yy[i:j]).sum(), (ww[i:j] * (1 - yy[i:j])).sum()
        tot += tp * (cumN + tn / 2); cumN += tn; i = j
    return tot / (P * N)

ones = np.ones(len(y)); case = y == 1
res = {"cycle": "NHANES August 2021-August 2023 (L files)",
       "flow": {"age_18_35_examined_with_MEC_weight": n_age, "excluded_known_diabetes": n_dx, "no_HbA1c": n_noa1c,
                "analytic": int(len(y)), "cases_HbA1c_ge_5.7": int(y.sum())},
       "pop_millions": w.sum() / 1e6, "prev": ratio(y), "cases_millions": (w * y).sum() / 1e6,
       "aware_among_cases": ratio(aware, case), "sens_uspstf": ratio(us, case),
       "sens_ada5": ratio((ada >= 5).astype(float), case & ~np.isnan(ada)),
       "model_at_dev_threshold": {"sens": ratio((p >= THR).astype(float), case), "flag_rate": ratio((p >= THR).astype(float))},
       "auc_unweighted": {"model": auc(p, y, ones), "ada": auc(ada, y, ones)}}

rng = np.random.default_rng(2026); groups = {}
for i, (s_, p_) in enumerate(zip(st, psu)): groups.setdefault(s_, {}).setdefault(p_, []).append(i)
diffs, am = [], []
for _ in range(500):
    idx = []
    for g in groups.values():
        ps = list(g.values())
        for k in rng.integers(0, len(ps), len(ps)): idx += ps[k]
    idx = np.array(idx); a2 = idx[~np.isnan(ada[idx])]
    diffs.append(auc(p[a2], y[a2], w[a2]) - auc(ada[a2], y[a2], w[a2])); am.append(auc(p[idx], y[idx], ones[idx]))
ma = ~np.isnan(ada)
res["auc_unweighted"]["model_ci_psu_bootstrap"] = list(np.percentile(am, [2.5, 97.5]))
res["auc_weighted"] = {"model": auc(p[ma], y[ma], w[ma]), "ada": auc(ada[ma], y[ma], w[ma]),
                       "diff": float(np.mean(diffs)), "diff_ci": list(np.percentile(diffs, [2.5, 97.5]))}

import statsmodels.api as sm
lg = np.log(p / (1 - p))
fit = sm.GLM(y, sm.add_constant(lg), family=sm.families.Binomial()).fit()
citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lg).fit().params[0]
res["calibration"] = {"slope": fit.params[1], "slope_ci": list(fit.conf_int()[1]), "citl": citl,
                      "mean_pred": p.mean(), "observed": y.mean(), "brier": ((p - y) ** 2).mean()}

tw, pw = w[ma].sum(), (w * y)[ma].sum(); o = np.argsort(-p[ma]); cw = np.cumsum(w[ma][o]); cy = np.cumsum((w * y)[ma][o])
res["matched_flag_rate"] = []
for c in [1, 2, 3, 4, 5]:
    f = ada[ma] >= c; fr = w[ma][f].sum() / tw
    k = np.searchsorted(cw / tw, fr)
    res["matched_flag_rate"].append({"ada_cut": c, "flag": fr, "ada": (w * y)[ma][f].sum() / pw, "model": cy[k - 1] / pw if k else 0.0})

conv = lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else v
json.dump(json.loads(json.dumps(res, default=conv)), open(os.path.join(os.path.dirname(__file__) or ".", "out", "temporal_2021_2023.json"), "w"), indent=1)
print(json.dumps(res, default=conv, indent=1))
