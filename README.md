# Credit Card Customer Churn Prediction

Predicting which credit-card customers are likely to leave (churn), using a Random Forest classifier on the BankChurners dataset.

## Overview

Retaining an existing customer is far cheaper than acquiring a new one, so being able to flag customers who are about to leave is directly useful to a bank. This project cleans the data, explores what actually drives churn, and trains a model to predict it — reporting the metrics that matter for an imbalanced problem rather than raw accuracy.

## Dataset

- **Source:** [BankChurners (Kaggle)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **Size:** 10,127 customers, 23 columns
- **Target:** `Attrition_Flag` — `Existing Customer` (0) vs `Attrited Customer` (1)
- **Class balance:** ~84% existing / ~16% churned (imbalanced — this shapes how the model is evaluated)

**Leakage removed:** the two `Naive_Bayes_Classifier_...` columns were dropped. They are the output of a model built *from the target*, so leaving them in would leak the answer and inflate results. `CLIENTNUM` was set as the index (identifier, no predictive value).

## Approach

**Cleaning & encoding**
- Dropped the leakage columns and the client identifier.
- Verified no null values and no impossible/negative entries.
- Encoded the target and `Gender` as binary (0/1).
- Used **ordinal encoding** for the naturally ordered categoricals (`Education_Level`, `Income_Category`, `Card_Category`, `Marital_Status`) — mapping each level to a number so the ranking is preserved. This suits a tree model, which only needs the order, not the spacing between levels.

**EDA**
- Pearson correlation (r) and p-values against the target to rank the numeric/binary features.
- Group-wise churn rates (`groupby` + `qcut` binning) to check for **non-linear** patterns that correlation misses, and to sanity-check small categories by group size.

**Modelling**
- Random Forest classifier, stratified 80/20 train/test split (stratified on the target to preserve the 84/16 balance).
- Trained two versions to test the feature-selection decision: one on the reduced feature set, one on all features (minus leakage).
- Confirmed the comparison with 5-fold cross-validation on recall.

## Key findings

- **Behaviour drives churn, not demographics.** The strongest signals are all *how the customer uses the card*: transaction count (`Total_Trans_Ct`, r = −0.37), change in transaction count Q4-vs-Q1 (r = −0.29), revolving balance (r = −0.26), and bank contacts (r = +0.20). Age, tenure, marital status, education, income and card tier were all effectively flat.

- **Two churn profiles.** Most signals are negative (declining activity → churn — the customer *fades out*), but bank-contact count is positive (more contact → churn — the customer is *dissatisfied*). Churners split into the disengaged and the frustrated.

- **A credit-limit threshold effect.** Correlation rated `Credit_Limit` as weak (r = −0.02), but binning revealed the lowest-limit quartile churns at ~20% vs ~14% for everyone else — a threshold, not a smooth trend. This is a genuine pattern that a linear correlation alone would have hidden, which is why the group-wise checks were worth doing.

- **Small groups mislead.** Apparent spikes in rare categories (e.g. Platinum cards) turned out to be artefacts of tiny sample sizes, not real signal — always checked group *size* alongside the rate.

## Results

Evaluated on the churn class (label 1), since **accuracy is misleading** here — a model predicting "nobody churns" would score ~84% while catching zero leavers.

| Model | Precision | Recall | F1 | 5-fold CV recall |
|---|:---:|:---:|:---:|:---:|
| Reduced features | 0.89 | 0.81 | 0.85 | 0.645 |
| All features (minus leakage) | 0.92 | 0.83 | 0.87 | 0.657 |

**Feature-selection finding:** several features with negligible *linear* correlation were initially dropped — but doing so slightly **reduced** recall. This indicates the Random Forest extracted weak *non-linear* signal from them that correlation couldn't see. The full feature set was kept, and cross-validation confirmed the small edge holds across folds rather than being one lucky split.

**On honest metrics:** single-split recall (~0.81–0.83) overstated performance. 5-fold cross-validation gave a more reliable ~0.65 — the figure to trust.

## How to run

```bash
git clone https://github.com/ryankong3573/Credit_Risk_Modelling.git
cd Credit_Risk_Modelling
pip install -r requirements.txt
jupyter notebook main.ipynb
```

Then run the notebook top to bottom.

## Data dictionary

| Column | Meaning |
|---|---|
| `Attrition_Flag` | Target — customer left (1) or stayed (0) |
| `Customer_Age` | Age in years |
| `Gender` | M / F |
| `Dependent_count` | Number of dependents |
| `Education_Level` | Highest education (Uneducated → Doctorate) |
| `Marital_Status` | Single / Married / Divorced / Unknown |
| `Income_Category` | Income band (<$40K → $120K+) |
| `Card_Category` | Card tier (Blue < Silver < Gold < Platinum) |
| `Months_on_book` | Length of relationship with the bank |
| `Total_Relationship_Count` | Number of products held with the bank |
| `Months_Inactive_12_mon` | Months inactive in the last year |
| `Contacts_Count_12_mon` | Times contacted the bank in the last year |
| `Credit_Limit` | Maximum credit available |
| `Total_Revolving_Bal` | Balance carried month to month |
| `Avg_Open_To_Buy` | Unused credit (`Credit_Limit − Revolving_Bal`) — dropped as redundant |
| `Total_Amt_Chng_Q4_Q1` | Change in spend amount, Q4 vs Q1 |
| `Total_Trans_Amt` | Total dollars spent (12 months) |
| `Total_Trans_Ct` | Total number of transactions (12 months) |
| `Total_Ct_Chng_Q4_Q1` | Change in transaction count, Q4 vs Q1 |
| `Avg_Utilization_Ratio` | Fraction of credit limit used |
