import json,sys,numpy as np,pandas as pd
sys.path.insert(0,'.')
from analysis import load, design, cv_predict, FEATS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm
DATA=sys.argv[1] if len(sys.argv)>1 else 'NHANES_data_2017-2018.tsv'
d,_=load(DATA); y=d.dysgly
lr=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2000))
out={}
for name,X in [('race_free',d[FEATS].copy()),('with_race',design(d))]:
    p=np.clip(cv_predict(lr,X,y),1e-6,1-1e-6); lp=np.log(p/(1-p))
    m=sm.GLM(y.values,sm.add_constant(lp),family=sm.families.Binomial()).fit()
    citl=sm.GLM(y.values,np.ones(len(y)),family=sm.families.Binomial(),offset=lp).fit()
    q=pd.qcut(p,10,labels=False,duplicates='drop')
    dec=[dict(mean_pred=float(p[q==k].mean()),observed=float(y.values[q==k].mean()),n=int((q==k).sum())) for k in sorted(set(q))]
    out[name]=dict(slope=float(m.params[1]),slope_ci=[float(x) for x in m.conf_int()[1]],intercept_in_large=float(citl.params[0]),
                   mean_pred=float(p.mean()),observed=float(y.mean()),brier=float(brier_score_loss(y,p)),deciles=dec)
json.dump(out,open('out/calibration.json','w'),indent=1)
for k,v in out.items(): print(k,{a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items() if a!='deciles'})
