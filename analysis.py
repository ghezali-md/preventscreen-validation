"""PreventScreen validation, pilot analysis on NHANES 2017-2018 (young adults, 18-35).

Question: in U.S. adults aged 18-35 without diagnosed diabetes, how much undiagnosed
dysglycemia (HbA1c >= 5.7%) is there, how much do current screening rules miss, and can a
model built only from routine data (anthropometrics, blood pressure, standard lipid panel,
family history) find it better than the ADA risk test?

Pilot limitations (addressed in the full protocol): single cycle, unweighted (survey weights
not in this extract), HbA1c-only outcome (no fasting glucose/insulin/triglycerides in this
extract), family history recorded only as present/blank.
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 2026

def load(path):
    d = pd.read_csv(path, sep="\t", low_memory=False)
    d = d[(d.AgeInYearsAtScreening >= 18) & (d.AgeInYearsAtScreening <= 35)].copy()
    n_age = len(d)
    diag = d.DoctorToldYouHaveDiabetes.astype(str).eq("1") | d.TakingInsulinNow.eq(1) | d.TakeDiabeticPillsToLowerBloodSugar.eq(1)
    d = d[~diag & d.Glycohemoglobin.notna()].copy()
    flow = dict(age_18_35=n_age, excluded_diagnosed_diabetes=int(diag.sum()), analytic=len(d))
    sbp = d[[f"SystolicBloodPres{k}MmHg" for k in ("1StRdg", "2NdRdg", "3RdRdg")]].mean(axis=1)
    dbp = d[[f"DiastolicBloodPres{k}MmHg" for k in ("1StRdg", "2NdRdg", "3RdRdg")]].mean(axis=1)
    d["sbp"], d["dbp"] = sbp, dbp
    d["male"] = d.Gender.eq("Male").astype(int)
    d["asian"] = d.RacehispanicOriginWNhAsian.eq("Non-Hispanic Asian").astype(int)
    d["fam_hx"] = d.FamilyHistory.eq("Family history").astype(int)
    act = d[["VigorousRecreationalActivities", "ModerateRecreationalActivities", "VigorousWorkActivity", "ModerateWorkActivity", "WalkOrBicycle"]]
    d["active"] = act.eq(1).any(axis=1).astype(int)
    d["htn"] = ((sbp >= 140) | (dbp >= 90)).astype(int)
    d["nonhdl"] = d.TotalCholesterolMgdl - d.DirectHdlcholesterolMgdl
    d["a1c"] = d.Glycohemoglobin
    d["dysgly"] = (d.a1c >= 5.7).astype(int)
    d["prediab"] = d.a1c.between(5.7, 6.4).astype(int)
    d["undx_dm"] = (d.a1c >= 6.5).astype(int)
    told = d.EverToldYouHavePrediabetes.eq(1) | d.DoctorToldYouHaveDiabetes.astype(str).eq("Borderline")
    d["aware"] = told.astype(int)
    return d, flow

def ada_score(d):
    """ADA Diabetes Risk Test (Bang et al., Ann Intern Med 2009), as implemented in PreventScreen.
    Gestational diabetes is not available in this extract and is scored 0."""
    age = np.select([d.AgeInYearsAtScreening < 40, d.AgeInYearsAtScreening < 50, d.AgeInYearsAtScreening < 60], [0, 1, 2], 3)
    b = d.BodyMassIndexKgm2
    cuts = np.where(d.asian == 1, 1, 0)  # Asian American cut-offs 23 / 27.5 / 37.5
    c1, c2, c3 = np.where(cuts, 23, 25), np.where(cuts, 27.5, 30), np.where(cuts, 37.5, 40)
    bmi = np.select([b < c1, b < c2, b < c3], [0, 1, 2], 3)
    bmi = np.where(b.isna(), np.nan, bmi)
    return age + d.male + d.fam_hx + d.htn + (1 - d.active) + bmi

FEATS = ["AgeInYearsAtScreening", "male", "BodyMassIndexKgm2", "WaistCircumferenceCm", "sbp", "dbp",
         "DirectHdlcholesterolMgdl", "nonhdl", "fam_hx", "active"]
RACE = ["Mexican American", "Other Hispanic", "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race - Including Multi-Racial"]

def design(d):
    X = d[FEATS].copy()
    for r in RACE:
        X["race_" + r.split()[0] + r.split()[-1]] = d.RacehispanicOriginWNhAsian.eq(r).astype(int)
    return X

def boot_ci(y, s, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); out = []
    y, s = np.asarray(y), np.asarray(s)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])

def cv_predict(model, X, y, reps=5):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=SEED)
    pred = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr]); pred[te] += model.predict_proba(X.iloc[te])[:, 1]; cnt[te] += 1
    return pred / cnt

def sens_spec(y, pos):
    y, pos = np.asarray(y), np.asarray(pos)
    return dict(sensitivity=float(pos[y == 1].mean()), specificity=float(1 - pos[y == 0].mean()), positive_rate=float(pos.mean()))

def main(path, outdir="."):
    d, flow = load(path)
    R = dict(flow=flow)
    R["prevalence"] = dict(n=len(d), dysglycemia=float(d.dysgly.mean()), prediabetes=float(d.prediab.mean()),
                           undiagnosed_diabetes=float(d.undx_dm.mean()),
                           aware_among_dysglycemia=float(d.loc[d.dysgly == 1, "aware"].mean()),
                           n_dysglycemia=int(d.dysgly.sum()))
    R["prevalence_by_bmi"] = {k: float(v) for k, v in d.groupby(pd.cut(d.BodyMassIndexKgm2, [0, 25, 30, 100], right=False, labels=["<25", "25-29.9", ">=30"]), observed=True).dysgly.mean().items()}
    d["ada"] = ada_score(d)
    m = d.ada.notna()
    R["ada_test"] = dict(n=int(m.sum()), auc=float(roc_auc_score(d.dysgly[m], d.ada[m])), auc_ci=list(boot_ci(d.dysgly[m], d.ada[m])),
                         at_threshold_5=sens_spec(d.dysgly[m], d.ada[m] >= 5))
    us = (d.AgeInYearsAtScreening >= 35) & (d.BodyMassIndexKgm2 >= 25)
    R["uspstf_2021"] = sens_spec(d.dysgly, us)
    X, y = design(d), d.dysgly
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30, random_state=SEED)
    p_lr, p_gb = cv_predict(lr, X, y), cv_predict(gb, X, y)
    for name, p in (("routine_logistic", p_lr), ("routine_boosting", p_gb)):
        thr = np.quantile(p, 1 - (d.ada[m] >= 5).mean())  # same positive rate as ADA >= 5
        R[name] = dict(auc=float(roc_auc_score(y, p)), auc_ci=list(boot_ci(y, p)), at_same_positive_rate_as_ada=sens_spec(y, p >= thr))
    # paired comparison on rows where ADA is available
    rng = np.random.default_rng(SEED); diffs = []
    ym, am, pm = d.dysgly[m].values, d.ada[m].values, p_lr[m.values]
    for _ in range(1000):
        i = rng.integers(0, len(ym), len(ym))
        if ym[i].min() != ym[i].max(): diffs.append(roc_auc_score(ym[i], pm[i]) - roc_auc_score(ym[i], am[i]))
    R["auc_difference_logistic_minus_ada"] = dict(mean=float(np.mean(diffs)), ci=list(np.percentile(diffs, [2.5, 97.5])))
    lr.fit(X, y)
    coefs = lr[-1].coef_[0]; R["logistic_standardized_coefficients"] = {c: float(v) for c, v in sorted(zip(X.columns, coefs), key=lambda t: -abs(t[1]))}
    json.dump(R, open(f"{outdir}/results.json", "w"), indent=1, default=float)
    # ROC figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, s, yy in (("ADA risk test", d.ada[m], d.dysgly[m]), ("Routine-data model (logistic)", p_lr, y), ("Routine-data model (boosting)", p_gb, y)):
        f, t, _ = roc_curve(yy, s); ax.plot(f, t, label=f"{lab} (AUC {roc_auc_score(yy, s):.2f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey"); ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Detecting HbA1c ≥ 5.7% in U.S. adults 18–35\n(NHANES 2017–2018, cross-validated)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(f"{outdir}/roc.png", dpi=200)
    return R

if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."), indent=1, default=float))

def extra(path, outdir="."):
    d, _ = load(path); d["ada"] = ada_score(d); m = d.ada.notna()
    R = json.load(open(f"{outdir}/results.json"))
    y = d.dysgly
    X_nr = d[FEATS].copy()  # race-free model
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    p_nr = cv_predict(lr, X_nr, y)
    X = design(d); p_lr = cv_predict(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)), X, y)
    R["routine_logistic_race_free"] = dict(auc=float(roc_auc_score(y, p_nr)), auc_ci=list(boot_ci(y, p_nr)))
    # ADA thresholds and model at matched sensitivity
    rows = []
    for t in range(1, 6):
        ss = sens_spec(d.dysgly[m], d.ada[m] >= t)
        rows.append(dict(threshold=t, **ss))
    R["ada_by_threshold"] = rows
    match = []
    for target in (0.5, 0.7, 0.8):
        out = dict(target_sensitivity=target)
        for name, p in (("routine_logistic", p_lr), ("routine_race_free", p_nr)):
            thr = np.quantile(p[y.values == 1], 1 - target)
            out[name] = sens_spec(y, p >= thr)
        # ADA: nearest threshold achieving >= target
        best = [r for r in rows if r["sensitivity"] >= target]
        out["ada_best_threshold_reaching_target"] = best[-1] if best else None
        match.append(out)
    R["matched_sensitivity"] = match
    # performance by group (race-free model)
    grp = {}
    for g, idx in d.groupby("RacehispanicOriginWNhAsian").groups.items():
        ii = d.index.get_indexer(idx); yy = y.values[ii]
        if yy.sum() >= 10 and (1 - yy).sum() >= 10:
            grp[g] = dict(n=int(len(ii)), cases=int(yy.sum()), auc_race_free=float(roc_auc_score(yy, p_nr[ii])))
    R["race_free_auc_by_group"] = grp
    sx = {}
    for g, idx in d.groupby("Gender").groups.items():
        ii = d.index.get_indexer(idx); sx[g] = dict(n=int(len(ii)), cases=int(y.values[ii].sum()), auc_race_free=float(roc_auc_score(y.values[ii], p_nr[ii])))
    R["race_free_auc_by_sex"] = sx
    json.dump(R, open(f"{outdir}/results.json", "w"), indent=1, default=float)
    return R

if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "extra":
    print(json.dumps({k: v for k, v in extra(sys.argv[1], sys.argv[2]).items() if k in ("routine_logistic_race_free", "ada_by_threshold", "matched_sensitivity", "race_free_auc_by_group", "race_free_auc_by_sex")}, indent=1, default=float))
