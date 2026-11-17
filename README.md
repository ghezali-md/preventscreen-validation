# preventscreen-validation

Pilot analysis for **PreventScreen**: how much undiagnosed dysglycemia (HbA1c ≥ 5.7%) exists in U.S. adults aged 18–35, how much current screening rules miss, and whether routine clinical data can find it better than the ADA risk test.

Author: Alla Eddine Ghezali, M.D.

## Key pilot results (NHANES 2017–2018, n = 1,416 aged 18–35 without diagnosed diabetes)
Survey-weighted national estimates (examination weights, Taylor-series linearization):
- 9.1% (95% CI 6.6–11.6%) had HbA1c ≥ 5.7% — about 6.6 million of 72 million U.S. young adults; only 15.6% of them had been told.
- USPSTF 2021 screening eligibility would flag 12.8% of them; the ADA risk test (score ≥ 5) 12.7%.
- Routine-data logistic model (including self-reported race and ethnicity): cross-validated AUC 0.76 (95% CI 0.73–0.80) vs. ADA risk test 0.69 (0.65–0.73); survey-weighted AUC difference 0.049 (95% CI 0.016–0.086).
- Race-free model used in PreventScreen (`out/model_race_free.json`): AUC 0.72 (0.68–0.75); calibration slope 0.87 (0.68–1.06), mean predicted 12.0% vs. observed 11.9%.

## Temporal validation (NHANES August 2021–August 2023, model frozen)
`python temporal.py DIR` (needs statsmodels) applies `out/model_race_free.json` unchanged to the official L-cycle files (DEMO_L, GHB_L, BMX_L, BPXO_L, HDL_L, TCHOL_L, DIQ_L, MCQ_L, PAQ_L). Results (`out/temporal_2021_2023.json`): n = 1,329 (122 cases); weighted prevalence of HbA1c ≥ 5.7% 9.2% (6.3–12.0%), about 6.5 million; 15.0% aware; sensitivity USPSTF 7.3%, ADA ≥ 5 5.0%; model AUC 0.69 (0.64–0.73) vs. ADA 0.65; weighted AUC difference 0.022 (−0.004 to 0.045, not significant); calibration slope 0.88 (0.63–1.13), mean predicted 10.5% vs. observed 9.2%. Family history (MCQ300C) is not released in this cycle and was set to 0 for everyone.

## Reproduce
```bash
pip install pandas numpy scikit-learn matplotlib nhanes==0.5.1
python analysis.py path/to/NHANES_data_2017-2018.tsv out     # main analysis -> out/results.json, out/roc.png
python -c "import analysis; analysis.extra('path/to/NHANES_data_2017-2018.tsv','out')"   # race-free model, matched sensitivity, subgroups
python weighted.py path/to/NHANES_data_2017-2018.tsv path/to/DEMO_J.xpt   # survey-weighted estimates -> out/weighted.json
python calib.py path/to/NHANES_data_2017-2018.tsv   # cross-validated calibration -> out/calibration.json
```
The data file ships inside the `nhanes` package (`nhanes/combined_data/2017-2018/`). `DEMO_J.xpt` (survey weights and design variables) is the official NCHS file: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt

## Limitations
Single cycle, models fit without weights (national estimates are weighted), HbA1c-only outcome, family history recorded as present/blank, internal cross-validation plus temporal validation on one later cycle (no family history in that cycle). The full protocol (pooled weighted cycles 2005–2023, fasting glucose/OGTT/HOMA-IR, temporal validation, calibration, fairness, TRIPOD+AI) is described in the accompanying report.

## License
MIT
