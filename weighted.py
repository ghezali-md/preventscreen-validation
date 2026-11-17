"""Survey-weighted estimates for the PreventScreen pilot (NHANES 2017-2018, MEC weights WTMEC2YR,
Taylor-series linearization with SDMVSTRA/SDMVPSU). Domain: ages 18-35, no diagnosed diabetes, HbA1c measured."""
import json, numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from analysis import load, ada_score, design, cv_predict, FEATS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

DATA = sys.argv[1] if len(sys.argv) > 1 else 'NHANES_data_2017-2018.tsv'
DEMO = sys.argv[2] if len(sys.argv) > 2 else 'DEMO_J.xpt'  # or DEMO_J.xpt from wwwn.cdc.gov
d, flow = load(DATA)
w = pd.read_sas(DEMO, format='xport')[['SEQN','WTMEC2YR','SDMVPSU','SDMVSTRA']] if DEMO.lower().endswith('.xpt') else pd.read_csv(DEMO)
w['SEQN'] = w.SEQN.astype(int)
d = d.merge(w, on='SEQN', how='left'); assert d.WTMEC2YR.notna().all()

def ratio_mean(y, x, wt, strata, psu, dom=None):
    """Weighted ratio sum(w*y)/sum(w*x) over domain, linearized SE (with-replacement PSU design)."""
    y=np.asarray(y,float); x=np.asarray(x,float); wt=np.asarray(wt,float)
    if dom is None: dom=np.ones_like(y,bool)
    y=np.where(dom,y,0); x=np.where(dom,x,0)
    R=(wt*y).sum()/(wt*x).sum()
    z=wt*(y-R*x)/(wt*x).sum()
    df=pd.DataFrame({'z':z,'s':strata,'p':psu}).groupby(['s','p']).z.sum().reset_index()
    var=0.0; dfree=0
    for s,g in df.groupby('s'):
        n=len(g); 
        if n<2: continue
        var+=n/(n-1)*((g.z-g.z.mean())**2).sum(); dfree+=n-1
    se=np.sqrt(var)
    from scipy.stats import t
    q=t.ppf(0.975, dfree)
    return R, se, (R-q*se, R+q*se), dfree

S,P,W=d.SDMVSTRA.values,d.SDMVPSU.values,d.WTMEC2YR.values
one=np.ones(len(d))
out={'n':len(d),'design_df':None,'population_millions':float(W.sum()/1e6)}
for k in ['dysgly','prediab','undx_dm']:
    R,se,ci,dfree=ratio_mean(d[k],one,W,S,P); out[k]=dict(p=R,ci=ci); out['design_df']=dfree
out['dysgly_count_millions']=dict(est=float((W*d.dysgly).sum()/1e6))
R,se,ci,_=ratio_mean(d.aware,one,W,S,P,dom=d.dysgly.eq(1).values); out['aware_among_dysgly']=dict(p=R,ci=ci)
for lab,cond in [('bmi_lt25',d.BodyMassIndexKgm2<25),('bmi_25_30',(d.BodyMassIndexKgm2>=25)&(d.BodyMassIndexKgm2<30)),('bmi_ge30',d.BodyMassIndexKgm2>=30)]:
    R,se,ci,_=ratio_mean(d.dysgly,one,W,S,P,dom=cond.fillna(False).values); out['dysgly_'+lab]=dict(p=R,ci=ci)
R,se,ci,_=ratio_mean((d.BodyMassIndexKgm2<25).astype(int),one,W,S,P,dom=d.dysgly.eq(1).values); out['share_of_cases_bmi_lt25']=dict(p=R,ci=ci)
c=d[d.dysgly==1]; Wc=c.WTMEC2YR
out['share_cases_not_obese']=float((Wc*(c.BodyMassIndexKgm2<30)).sum()/Wc[c.BodyMassIndexKgm2.notna()].sum())
out['share_cases_insured']=float((Wc*c.CoveredByHealthInsurance.eq(1)).sum()/Wc.sum())
# screening rules (weighted sensitivity among cases)
ada=ada_score(d); d['ada5']=(ada>=5).astype(float); d.loc[ada.isna(),'ada5']=np.nan
uspstf=(d.AgeInYearsAtScreening>=35)&(d.BodyMassIndexKgm2>=25)  # same rule as the pilot
d['uspstf']=uspstf.astype(int)
for k in ['ada5','uspstf']:
    m=d[k].notna().values & d.dysgly.eq(1).values
    R,se,ci,_=ratio_mean(d[k].fillna(0),one,W,S,P,dom=m); out['sens_'+k]=dict(p=R,ci=ci)
# weighted AUCs of cross-validated predictions (models fit unweighted, as in the pilot)
X=design(d); y=d.dysgly
lr=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2000))
p=cv_predict(lr,X,y); y=y.values
ok=ada.notna().values
out['auc_weighted']=dict(routine_logistic=roc_auc_score(y,p,sample_weight=W),ada=roc_auc_score(y[ok],ada[ok],sample_weight=W[ok]))
Xr=d[FEATS].copy(); pr=cv_predict(lr,Xr,d.dysgly); out['auc_weighted']['race_free']=roc_auc_score(y,pr,sample_weight=W)
# weighted bootstrap of AUC difference by resampling PSUs within strata
rng=np.random.default_rng(2026); diffs=[]
grp={s:[np.where((S==s)&(P==q))[0] for q in np.unique(P[S==s])] for s in np.unique(S)}
for b in range(500):
    idx=np.concatenate([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]) for g in grp.values()])
    o=ok[idx]
    try: diffs.append(roc_auc_score(y[idx][o],p[idx][o],sample_weight=W[idx][o])-roc_auc_score(y[idx][o],ada.values[idx][o],sample_weight=W[idx][o]))
    except ValueError: pass
out['auc_diff_weighted']=dict(mean=float(np.mean(diffs)),ci=[float(x) for x in np.percentile(diffs,[2.5,97.5])],n_boot=len(diffs))
import os; os.makedirs('out',exist_ok=True); json.dump(out,open('out/weighted.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=lambda v: round(float(v),4)))
