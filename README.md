# IEEE-CIS-Fraud-Detection-ML2
### Kaggle-ის კონკურსის მიმოხილვა
IEEE-CIS Fraud Detection - კონკურსის მიზანია ისეთი მოდელის შექმნა, რომეიც მოცემული დატასეტის მიხედვით ტესტზე სწორ პროგნოზს გააკეთებს. 
ამ დავალებაში გამოვიყენე მთავარ მეტრიკად ROC-AUC, თუმცა ვინაიდან დაუბალანსებელი დატა იყო(class imbalance ~3.5%), ვაკვირდებოდი PR-AUC-საც. 

## რეპოზიტორიის სტრუქტურა
 
```
IEEE-CIS-Fraud-Detection-ML2/
│
├── experiment_XGBoost.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_AdaBoost.ipynb
├── model_inference.ipynb
└── README.md
```

### ფაილების აღწერა
 
| ფაილი | აღწერა |
|-------|--------|
| `model_experiment_XGBoost.ipynb` | XGBoost — WOE+OHE+Freq encoding, IV selection, training|
| `model_experiment_RandomForest.ipynb` | Random Forest — median imputation, RFE, Gini importance, training|
| `model_experiment_AdaBoost.ipynb` | AdaBoost — DecisionTree base, outlier capping, training |
| `model_inference.ipynb` | Registry-დან საუკეთესო მოდელის pipeline-ს ჩამოტვირთვა, submission ფაილი|
 
---

### Feature Engineering
მონაცემების წინასწარ მოსამზადებლად პირველ რიგში წავშალე ისეთი სვეტები, რომელსაც 90%ზე მეტი NaN მნიშვნელობა ჰქონდა. ასევე დავამატე ახალი სვეტები, რომლების მიანიშვნებდა, რომ ესათუ ის სვეტი შეიცავდა NaN-ებს(missing cols - _was_nan).  შემდეგ გამოვყავი სვეტები კარდინალობების მიხედვით. high cardinality სვეტებზე (რომელიც 50ზე მეტი განსხვავებული მნიშვნელობას იღებდა), გამოვიყენე frequency encoding, ხოლო low cardinality სვეტებზე label encoding. ცარიელი ადგილები (NaN) შევავსე მედიანური მნიშვნელობით, რათა mean მნიშვნელობის შემთხვევაში outlier-ებს არ გამოეწვია შეცდომა. გამოვიყენე მხოლოდ x_train-ის მონაცემები და ამით შევავსე ვალიდაციის ხვეტებიც. (ანუ .fit.transform() x_train-ზე, .transform() x_valid-ზე). 
ასევე გამოვიყენე outlier clipping მეთოდი, რომლითაც extreme value-ები, რომლებიც საშუალო მონაცემების საზღვრებს სცდებოდა შევცვალე მინიმალური და მაქსიმალური მნიშვნელობეთ. ძალიან მსგავსი feature-ების დუპლიკაციისგან თავის ასარიდებლად გამოვიყენე კორელაციის ფილტრი 0.95 threshold-თ. აგრეთვე ვარიაციის ფილტრი მცირე ვარიაციის მქონე სვეტების წასაშლელად (Threshold = 0.01).
მიუხედავად იმისა, რომ xgboost-ს გააჩნია NaN მნიშვნელობებთან გამკლავების უნარი, თავიდანვე ყოველი მოდელისთის გავაკეთე ერთი და იგივე cleaning მეთოდები.

ბოლოს, დავამატე რამდენიმე ისეთი feature, რომლებიც უფრო მეტად გაუმარტივებს მოდელს საქმეს. მაგ:'hour_of_day', 'is_weekend', 'is_night', 'amt_is_round', ვინაიდან fraud ტრანზაქციები ხდება შაბათ-კვირის პერიოდში, უფრო მეტად ღამით, და მრგვალი/მთელი რიცხვი შეიძლება უფრო მეტად იყოს დაკავშირებული მასთან.

### Feature Selection
randomForest -ის შემთხვევაში feature selection-ს მეთოდად ვცადე ორივე ცალცალკე: gini importance და RFE. შევადარე მათი შედეგები და საბოლოოდ ავირჩიე RFE, რადგან მისი საშუალებით საბოლოოდ დარჩა 120 feature და validation AUC უფრო მეტი ჰქონდა ამ ეტაპისთვის.
<Figure size 1200x400 with 2 Axes>


### Training
ტესტირებული მოდელები:

## Training
 
### XGBoost (საუკეთესო)
 
**მახასიათებლები:**
- Native NaN handling (no imputation)
- GPU acceleration (`device='cuda'`, `tree_method='hist'`)
- `eval_metric='aucpr'` — PR-AUC fraud imbalance-ისთვის უფრო informative
- `scale_pos_weight = (neg_samples / pos_samples)` — class imbalance
- Early stopping (`early_stopping_rounds=100`)
- 
**Grid Search:**
 
| კონფიგურაცია | CV PR-AUC | Val AUC | Gap | სტატუსი |
|---------|-----------|---------|-----|---------|
| depth=4, n=100, lr=0.1 | 0.46 | 0.89 | 0.02 | OK |
| depth=6, n=1500, lr=0.05 | 0.75 | 0.93 | 0.02 | OK |
| depth=6, n=3000, lr=0.01 | 0.69 | 0.92 | 0.01 | OK |
| depth=8, n=1500, lr=0.05 | 0.83 | 0.93 | 0.03 | OK |
| depth=6, reg_a=1.0, reg_l=5 | 0.76 | 0.92 | 0.01 | OK |
| depth=10, n=1500, lr=0.1 | 0.85 | 0.93 | 0.04 | Overfit |
| **depth=12, n=1500, lr=0.05** | **0.837** | **0.96** | **0.04** | **Best** |

 
**საბოლოო Hyperparameters:**
 
| პარამეტრი | მნიშვნელობა |
|-----------|------------|
| `n_estimators` | 1500 |
| `max_depth` | 12 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.7 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `min_child_weight` | 10 |
| `eval_metric` | aucpr |
| `tree_method` | hist |
| `device` | cuda |
| `random_state` | 42 |
 
**Overfitting ანალიზი:**
- `depth=10, lr=0.1, n=1500` — gap=0.04, borderline overfit: high lr + deep trees → fast overfitting
- `depth=12, min_child_weight=10` — regularization balances depth: min_child_weight=10 prevents leaf splits with few samples → acceptable gap

### Random Forest
 
**მახასიათებლები:**
- Median imputation 
- `class_weight='balanced'` fraud imbalance-ისთვის

 
| კონფიგურაცია | Val AUC | Val PR-AUC | Gap | სტატუსი |
|---------|---------|-----------|-----|---------|
| depth=6, min_leaf=30, n=200 | 0.87 | 0.52 | 0.06 | OK |
| depth=10, min_leaf=20, n=150 | 0.89 | 0.55 | 0.07 | OK |
| depth=12, min_leaf=15, n=200 | 0.90 | 0.57 | 0.08 | Overfit |
| **depth=15, min_leaf=10, n=200** | **0.91** | **0.58** | **0.09** | ** Best** |
 
**საბოლოო Hyperparameters:**
 
| პარამეტრი | მნიშვნელობა |
|-----------|------------|
| `n_estimators` | 250 |
| `max_depth` | 15 |
| `min_samples_leaf` | 20 |
| `max_features` | sqrt |
| `class_weight` | balanced |
| `oob_score` | True |
| `random_state` | 42 |
 
**Overfitting ანალიზი:**
- `max_depth=None, min_samples_leaf=1` → Tree-ები train data-ს ზეპირად ისწავლის (train AUC ~0.999, gap ~0.09)
- `max_depth=3, min_samples_leaf=100` → ძალიან პატარა ხეები, პატერნებს ვერ ასახავს (underfit)
- `max_depth=15 + min_samples_leaf=20` — რეგულარიზაცია კომპლექსურობას ამცირებს
---
 
### AdaBoost
 
**მახასიათებლები:**
- Outlier capping — weight explosion prevention
- Imputation  — DecisionTree NaN-ს ვერ ამუშავებს
- WOE encoding — fraud signal-ი base learner-ს უადვილებს
- `algorithm='SAMME'` — SAMME.R deprecated sklearn-ში
**Grid Search:**
 
| კონფიგურაცია | CV AUC | Val AUC | Gap | სტატუსი |
|---------|--------|---------|-----|---------|
| depth=1, n=300, lr=1.0 | 0.85 | 0.86 | 0.01 | OK |
| depth=1, n=500, lr=0.5 | 0.86 | 0.87 | 0.01 | OK |
| depth=1, n=500, lr=0.1 | 0.83 | 0.84 | 0.01 | Underfit |
| depth=2, n=300, lr=1.0 | 0.87 | 0.87 | 0.02 | OK |
| **depth=2, n=500, lr=0.5** | **0.88** | **0.88** | **0.02** | **Best** |
| depth=3, n=200, lr=0.5 | 0.89 | 0.86 | 0.06 | Overfit |
| depth=3, n=1000, lr=1.0 | 0.93 | 0.84 | 0.14 | Overfit |
 
**Overfitting ანალიზი:**
- `depth=3, n=1000, lr=1.0` — base learner ძალიან კომპლექსურია (depth=3) + ძალიან ბევრი იტერაციაა და ვარიაცია მაღალია, gap=0.14
- `depth=1, lr=0.01, n=10` —  weak learners + low learning rate → მოდელი ვერ სწავლობს ამიტომ არის underfit
- `depth=2 + lr=0.5 + n=500` 
---
 
ყველა კონფიგურაციაზე გამოვიყენე 5-Fold Stratified Cross Validation. xgboost-ში გამოვიყენე early stopping.
 
---
 
## საბოლოო მოდელის შერჩევა
 
| მოდელი | Val ROC-AUC | Val PR-AUC |
|--------|------------|-----------|
| AdaBoost | 0.880 | 0.51 |
| Random Forest | 0.912 | 0.58 |
| **XGBoost** | **0.940** | **0.837** |
 
**XGBoost**:
- ყველაზე მაღალი Val AUC და PR-AUC
- `eval_metric='aucpr'` — imbalanced data-ზე უფრო ინფორმატიული მეტრიკაა და xgboost-ის შემთხვევაში სხვებთან შედარებით საკმაოდ მაღალია.
- ყველაზე სწრაფი მოდელიც იყო xgboost.


MLflow Tracking
MLflow ექსპერიმენტების ბმული:
https://dagshub.com/akave23/IEEE-CIS-Fraud-Detection-ML2/experiments
ჩაწერილი მეტრიკების აღწერა
### ჩაწერილი მეტრიკები
 
| მეტრიკა | აღწერა |
|---------|--------|
| `auc_val_holdout` | Held-out X_valid ROC-AUC |
| `prauc_val` | Held-out X_valid PR-AUC |
| `auc_cv_mean` | 5-Fold CV საშუალო ROC-AUC |
| `auc_cv_std` | CV AUC სტანდარტული გადახრა |
| `auc_train_avg` | Train AUC (fold average) |
| `overfitting_gap` | train_auc - cv_auc |
| `features_final` | Feature selection შემდეგ |
 

საუკეთესო მოდელის შედეგები
## საუკეთესო მოდელის შედეგები (XGBoost)

| მეტრიკა | მნიშვნელობა |
|---------|------------|
| auc_val | ~0.971 |
| Val PR-AUC | ~0.865 |
| Train AUC | ~0.9997 |
| Overfitting Gap | ~0.0371 |

 
<img width="1367" height="107" alt="image" src="https://github.com/user-attachments/assets/fb7e4635-403f-4358-936c-3973e1903311" />
