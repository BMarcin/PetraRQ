
Eur-lex-documents
=============================

Eur-lex-documents multilabel long documents classification.
Assign one, more than one or none labels to each doc.

Documents were downloaded from [European Commision's website](https://eur-lex.europa.eu/browse/institutions/eu-commission.html).

Dataset labels
-------------------
- agriculture
- work_and_employment
- education
- taxes
- industry
- european_union
- law
- state_public_authorities
- economy
- environment
- foreign_policy
- science_research_and_technology
- health
- sports
- social_policy
- transport

Repository source code
-------------------
[https://github.com/BMarcin/PetraRQ](https://github.com/BMarcin/PetraRQ)

Pretrained models
---
To switch between different pretrained models use git branching & dvc pulling. You can switch branch to one from listed below:
- models/fasttext-books - **FastText** model pretrained on full **Books** dataset
- models/petrarq-books - **PetraRQ** model pretrained on full **Books** dataset
- models/petrarq-10k - **PetraRQ** model pretrained on **EurLex-10k** dataset
- models/petrarq-1k - **PetraRQ** model pretrained on **EurLex-1k** dataset
- models/petrarq-100 - **PetraRQ** model pretrained on **EurLex-100** dataset
- models/petrarq - **PetraRQ** model pretrained on **EurLex** dataset
- models/fasttext-10k - **FastText** model pretrained on **EurLex-10k** dataset
- models/fasttext-1k - **FastText** model pretrained on **EurLex-1k** dataset
- models/fasttext-100 - **FastText** model pretrained on **EurLex-100** dataset
- models/fasttext - **FastText** model pretrained on **EurLex** dataset
- models/xgboost-10k - **XGBoost** model pretrained on **EurLex-10k** dataset
- models/xgboost-1k - **XGBoost** model pretrained on **EurLex-1k** dataset
- models/xgboost-100 - **XGBoost** model pretrained on **EurLex-100** dataset 
- models/xgboost - **XGBoost** model pretrained on **EurLex** dataset
- models/random_forest-10k - **Random Forest** model pretrained on **EurLex-10k** dataset
- models/random_forest-1k - **Random Forest** model pretrained on **EurLex-1k** dataset
- models/random_forest-100 - **Random Forest** model pretrained on **EurLex-100** dataset
- models/random_forest - **Random Forest** model pretrained on **EurLex** dataset
- models/svm-10k - **SVM/SVC** model pretrained on **EurLex-10k** dataset
- models/svm-1k - **SVM/SVC** model pretrained on **EurLex-1k** dataset
- models/svm-100 - **SVM/SVC** model pretrained on **EurLex-100** dataset
- models/svm - **SVM/SVC** model pretrained on **EurLex** dataset
- models/naive_bayes-10k - **Naive Bayes** model pretrained on **EurLex-10k** dataset
- models/naive_bayes-1k - **Naive Bayes** model pretrained on **EurLex-1k** dataset
- models/naive_bayes-100 - **Naive Bayes** model pretrained on **EurLex-100** dataset
- models/naive_bayes - **Naive Bayes** model pretrained on **EurLex** dataset
- models/logistic_regression-10k - **Logistic Regression** model pretrained on **EurLex-10k** dataset
- models/logistic_regression-1k - **Logistic Regression** model pretrained on **EurLex-1k** dataset
- models/logistic_regression-100 - **Logistic Regression** model pretrained on **EurLex-100** dataset
- models/logistic_regression - **Logistic Regression** model pretrained on **EurLex** dataset
- models/roberta-10k - **RoBERTa** model pretrained on **EurLex-10k** dataset
- models/roberta-1k - **RoBERTa** model pretrained on **EurLex-1k** dataset
- models/roberta-100 - **RoBERTa** model pretrained on **EurLex-100** dataset
- models/roberta - **RoBERTa** model pretrained on **EurLex** dataset

Pulling models/datasets
---
```bash
dvc pull # <- pull all data listed in dvc.yaml
dvc pull ./data/parsed-pdfs.json # <- pull parsed-pdfs.json file
```

Retraining
---
```bash
make # <- it will pull needed data and retrain models using DVC pipelines
# or after pulling data manually
dvc repro
```

Prediting
---
```bash
./scripts/predict.sh <string to predict>
```

Environment
---
Install all packages listed in `requirements.txt` file
```bash
pip install -r requirements.txt
```

or use it with builded docker. List of available docker tags can be found [here](https://github.com/BMarcin/PetraRQ/pkgs/container/petrarq/versions).