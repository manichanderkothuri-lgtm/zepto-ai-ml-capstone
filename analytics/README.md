# Module 2 — Analytics + Machine Learning

Run `python analysis.py`. The script loads Seaborn's Titanic dataset once, immediately saves `titanic.csv`, performs the required EDA/cleaning, creates the required charts and reports, trains Logistic Regression, Decision Tree and Random Forest on the same stratified split, compares imbalance strategies, tunes Random Forest with GridSearchCV and OOB scoring, performs fare regression, saves the best complete classifier pipeline with joblib, and verifies reloading it on raw input.

The EDA cleaning rule is applied explicitly: <5% missing → drop affected rows; 5–30% → median/mode imputation; columns with >30% missing that are not reliable for imputation are dropped (`deck`). Modeling preprocessing is independently fit only on the training split through a scikit-learn ColumnTransformer/Pipeline.

Each generated chart has a corresponding interpretation in `interpretations.md`; numeric results are written to CSV/JSON files.
