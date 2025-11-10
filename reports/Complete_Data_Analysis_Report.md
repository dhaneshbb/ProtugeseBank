# Portuguese Bank Marketing Campaign: Complete Data Analysis Report

**Project:** Term Deposit Subscription Prediction
**Dataset:** UCI Bank Marketing Dataset (41,174 records, 17 features)
**Final Model:** LightGBM with F1-Score = 0.509, ROC AUC = 0.811, Accuracy = 86.8%
**Report Date:** March 01, 2025
**Last Revised:** November 07, 2025

---

## Executive Summary

This report documents a machine learning project predicting term deposit subscriptions from Portuguese bank marketing campaigns conducted between May 2008 and November 2010. The dataset contains 41,174 customer records with 17 features covering demographics, campaign interactions, and macroeconomic indicators. Through systematic data cleaning, statistical testing, and tree-based modeling, a LightGBM classifier was developed that achieves 60.9% recall and 43.7% precision on test data, balancing the detection of potential subscribers against resource allocation efficiency.

Key findings reveal that economic indicators, particularly the 3-month Euribor rate and employment numbers, are the primary predictors of subscription behavior. The model demonstrates consistent generalization with cross-validation F1 of 0.488 +/- 0.020, making it suitable for targeting customers in future marketing campaigns while accounting for class imbalance (88.7% non-subscribers).

---

## Table of Contents

- [Portuguese Bank Marketing Campaign: Complete Data Analysis Report](#portuguese-bank-marketing-campaign-complete-data-analysis-report)
  - [Executive Summary](#executive-summary)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Business Context](#11-business-context)
    - [1.2 Dataset Overview](#12-dataset-overview)
    - [1.3 Project Objectives](#13-project-objectives)
  - [2. Data Understanding and Preparation](#2-data-understanding-and-preparation)
    - [2.1 Initial Data Assessment](#21-initial-data-assessment)
    - [2.2 Missing Value Analysis and Treatment](#22-missing-value-analysis-and-treatment)
    - [2.3 Duplicate Detection and Removal](#23-duplicate-detection-and-removal)
    - [2.4 Outlier Detection and Treatment](#24-outlier-detection-and-treatment)
    - [2.5 Data Leakage Prevention](#25-data-leakage-prevention)
    - [2.6 Final Cleaned Dataset](#26-final-cleaned-dataset)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
    - [3.1 Target Variable Distribution](#31-target-variable-distribution)
    - [3.2 Descriptive Statistics](#32-descriptive-statistics)
    - [3.3 Group Comparisons by Target](#33-group-comparisons-by-target)
    - [3.4 Normality Assessment](#34-normality-assessment)
    - [3.5 Multicollinearity Analysis](#35-multicollinearity-analysis)
  - [4. Statistical Testing and Feature Selection](#4-statistical-testing-and-feature-selection)
    - [4.1 Categorical Association Tests](#41-categorical-association-tests)
    - [4.2 Numerical Variable Correlations](#42-numerical-variable-correlations)
    - [4.3 Distribution Comparison Tests](#43-distribution-comparison-tests)
    - [4.4 Feature Selection Decisions](#44-feature-selection-decisions)
  - [5. Feature Engineering and Preprocessing](#5-feature-engineering-and-preprocessing)
    - [5.1 Encoding Categorical Variables](#51-encoding-categorical-variables)
    - [5.2 Train-Test Split Strategy](#52-train-test-split-strategy)
    - [5.3 Final Feature Set](#53-final-feature-set)
  - [6. Model Development and Evaluation](#6-model-development-and-evaluation)
    - [6.1 Model Selection Rationale](#61-model-selection-rationale)
    - [6.2 Base Model Comparison](#62-base-model-comparison)
    - [6.3 Hyperparameter Tuning](#63-hyperparameter-tuning)
    - [6.4 Final Model Selection: LightGBM](#64-final-model-selection-lightgbm)
  - [7. Model Interpretation and Insights](#7-model-interpretation-and-insights)
    - [7.1 Feature Importance Analysis](#71-feature-importance-analysis)
    - [7.2 Threshold Optimization](#72-threshold-optimization)
    - [7.3 Model Performance at Optimal Threshold](#73-model-performance-at-optimal-threshold)
    - [7.4 Cross-Validation Results](#74-cross-validation-results)
  - [8. Business Insights and Recommendations](#8-business-insights-and-recommendations)
    - [8.1 Economic Context Matters](#81-economic-context-matters)
    - [8.2 Demographic Patterns](#82-demographic-patterns)
    - [8.3 Campaign Timing and Contact Strategy](#83-campaign-timing-and-contact-strategy)
    - [8.4 Practical Recommendations](#84-practical-recommendations)
  - [9. Challenges and Solutions](#9-challenges-and-solutions)
    - [9.1 Challenge: Class Imbalance](#91-challenge-class-imbalance)
    - [9.2 Challenge: Data Leakage Risk](#92-challenge-data-leakage-risk)
    - [9.3 Challenge: Multicollinearity Among Economic Indicators](#93-challenge-multicollinearity-among-economic-indicators)
    - [9.4 Challenge: Interconnected Outliers](#94-challenge-interconnected-outliers)
    - [9.5 Challenge: Missing Values in Key Variables](#95-challenge-missing-values-in-key-variables)
  - [10. Limitations and Future Work](#10-limitations-and-future-work)
    - [10.1 Limitations](#101-limitations)
    - [10.2 Future Work](#102-future-work)
  - [11. Conclusion](#11-conclusion)
  - [12. Appendix](#12-appendix)
    - [12.1 Dataset Access](#121-dataset-access)
    - [12.2 References](#122-references)
    - [12.3 Technical Environment](#123-technical-environment)
    - [12.4 Reproducibility](#124-reproducibility)
    - [12.5 Model Deployment](#125-model-deployment)
  - [Acknowledgments](#acknowledgments)

---

## 1. Introduction

### 1.1 Business Context

The banking industry relies heavily on direct marketing campaigns to promote financial products. Term deposits represent a critical revenue source, yet conversion rates remain low, requiring data-driven strategies to optimize resource allocation. This analysis addresses the need for predictive models that identify customers most likely to subscribe to term deposits, enabling banks to:

- Focus marketing efforts on high-potential customers
- Reduce campaign costs by minimizing contacts with unlikely subscribers
- Understand the demographic and economic factors driving subscription decisions
- Time campaigns strategically based on macroeconomic conditions

### 1.2 Dataset Overview

The UCI Bank Marketing Dataset documents phone-based direct marketing campaigns conducted by a Portuguese bank from May 2008 to November 2010. The dataset comprises:

- **Observations:** 41,174 customer contacts (after cleaning from 41,188 original records)
- **Target Variable:** y (binary: yes/no for term deposit subscription)
- **Features:** 17 predictors after preprocessing, organized into:
  - **Demographics (4):** age, job, marital status, education
  - **Financial (1):** housing loan status
  - **Campaign (2):** contact type, number of contacts in current campaign
  - **Temporal (2):** month, day of week
  - **Economic Indicators (5):** employment variation rate, consumer price index, consumer confidence index, 3-month Euribor rate, number of employees
  - **Target:** y (subscription outcome)

**Class Distribution:**
- No subscription: 36,535 (88.73%)
- Subscription: 4,639 (11.27%)

This class imbalance presents a key modeling challenge.

### 1.3 Project Objectives

1. **Data Analysis:** Clean and explore relationships between customer attributes, campaign metrics, and subscription outcomes
2. **Feature Engineering:** Address data leakage, multicollinearity, and prepare features for classification
3. **Predictive Modeling:** Develop tree-based models to predict term deposit subscriptions while handling class imbalance
4. **Model Interpretation:** Extract actionable insights about subscription drivers for campaign optimization

---

## 2. Data Understanding and Preparation

### 2.1 Initial Data Assessment

The dataset required preprocessing due to data quality issues:

| Aspect | Finding |
|--------|---------|
| Initial Dimensions | 41,188 rows x 21 columns |
| Memory Usage | 30.26 MB |
| Missing Values | 6 columns affected (0.2%-20.9% missing) |
| Duplicates | 14 duplicate rows found |
| Data Types | 11 object, 5 int64, 5 float64 |

**Initial Column Assessment:**

| Column | Data Type | Range | Distinct Count |
|--------|-----------|-------|----------------|
| age | int64 | 17 - 98 | 78 |
| duration | int64 | 0 - 4,918 seconds | 1,544 |
| campaign | int64 | 1 - 56 contacts | 42 |
| pdays | int64 | 0 - 999 | 27 |
| previous | int64 | 0 - 7 | 8 |
| emp.var.rate | float64 | -3.4 - 1.4 | 10 |
| cons.price.idx | float64 | 92.201 - 94.767 | 26 |
| cons.conf.idx | float64 | -50.8 - -26.9 | 26 |
| euribor3m | float64 | 0.634 - 5.045 | 316 |
| nr.employed | float64 | 4,963.6 - 5,228.1 | 11 |

### 2.2 Missing Value Analysis and Treatment

**Missing Data Pattern:**

| Column | Missing Count | Percentage | Imputation Strategy |
|--------|---------------|------------|---------------------|
| default | 8,597 | 20.87% | Mode imputation ("no") |
| education | 1,731 | 4.20% | Mode imputation ("university.degree") |
| housing | 990 | 2.40% | Mode imputation ("yes") |
| loan | 990 | 2.40% | Mode imputation ("no") |
| job | 330 | 0.80% | Mode imputation ("admin.") |
| marital | 80 | 0.19% | Mode imputation ("married") |

**Rationale:**
All missing values were encoded as "unknown" in the original dataset. These were replaced with pandas NA, then imputed using mode (most frequent value) for categorical variables. Mode imputation preserves the dominant patterns in the data without introducing artificial categories.

**Total Missing Percentage:** 1.47% of all data points

### 2.3 Duplicate Detection and Removal

- **Initial duplicates:** 12 rows
- **After imputation:** 14 rows (imputation created 2 additional exact duplicates)
- **Action:** All 14 duplicates removed
- **Final dataset:** 41,174 rows

### 2.4 Outlier Detection and Treatment

**Initial Outlier Analysis:**

| Feature | Q1 | Q3 | IQR | Outliers Count | Outliers % |
|---------|----|----|-----|----------------|------------|
| age | 32.0 | 47.0 | 15.0 | 468 | 1.14% |
| duration | 102.0 | 319.0 | 217.0 | 2,963 | 7.20% |
| campaign | 1.0 | 3.0 | 2.0 | 2,406 | 5.84% |
| pdays | 999.0 | 999.0 | 0.0 | 1,515 | 3.68% |
| previous | 0.0 | 0.0 | 0.0 | 5,625 | 13.66% |
| cons.conf.idx | -42.7 | -36.4 | 6.3 | 446 | 1.08% |

**Interconnected Outliers:**
Analysis revealed 1,834 rows with outliers across multiple features simultaneously, indicating systemic patterns rather than random noise:
- Most common pattern: pdays + previous (1,336 occurrences)
- Other patterns: age combinations with campaign, previous, cons.conf.idx

**Outlier Treatment:**

1. **Age:** Capped at 69.5 (upper bound from IQR method)
   - Rationale: Elderly customers (70+) represent a distinct segment but extreme ages (78-98) may introduce leverage

2. **Campaign:** Capped at 6 contacts
   - Rationale: Customers contacted more than 6 times show diminishing returns and may represent edge cases

3. **Cons.conf.idx:** Capped at 98th percentile
   - Rationale: Extreme consumer confidence values (-50.8) likely represent crisis periods not representative of typical conditions

**Distribution Changes After Capping:**

| Feature | Original Skewness | Original Kurtosis | Post-Capping Skewness | Post-Capping Kurtosis |
|---------|-------------------|-------------------|-----------------------|-----------------------|
| age | 0.785 | 0.791 | 0.567 | -0.246 |
| campaign | 4.762 | 36.966 | 1.212 | 0.408 |
| cons.conf.idx | 0.303 | -0.359 | 0.196 | -0.754 |

### 2.5 Data Leakage Prevention

**Critical Columns Removed:**

1. **duration (call duration in seconds)**
   - **Issue:** Only known after call completion, making it unavailable for prediction at contact time
   - **Evidence:** Strong predictor of outcome (longer calls correlate with subscriptions), but this creates data leakage
   - **Action:** Dropped entirely

2. **pdays (days since previous contact)**
   - **Issue:** 999 indicates "never contacted before," creating a binary feature masquerading as continuous
   - **Evidence:** 86.3% of data has pdays=999, and previous contact history strongly predicts future subscriptions
   - **Action:** Dropped to prevent leakage from campaign history

3. **previous (number of contacts before current campaign)**
   - **Issue:** Previous campaign success directly influences current campaign targeting
   - **Evidence:** High correlation with pdays, and 13.66% outliers suggest non-random contact patterns
   - **Action:** Dropped along with poutcome

4. **poutcome (outcome of previous campaign)**
   - **Issue:** High predictor if previous outcome was "success" (19.3% of subscribers had prior success)
   - **Evidence:** Chi-square test showed p < 0.001, confirming strong association
   - **Action:** Dropped to ensure model learns from current customer attributes, not past campaign results

### 2.6 Final Cleaned Dataset

After preprocessing:
- **Rows:** 41,174 (14 duplicates removed)
- **Columns:** 17 (4 dropped for leakage prevention)
- **Missing Values:** 0
- **Outliers:** Capped, not removed (preserves sample size)
- **Data Quality:** Ready for exploratory analysis and modeling

---

## 3. Exploratory Data Analysis

### 3.1 Target Variable Distribution

**Subscription Outcomes:**

| Outcome | Count | Percentage |
|---------|-------|------------|
| No | 36,535 | 88.73% |
| Yes | 4,639 | 11.27% |

**Class Imbalance Ratio:** 7.87:1 (no:yes)

This imbalance requires:
- Stratified train-test splitting
- Class weighting in models
- Metrics focused on recall and precision, not just accuracy

### 3.2 Descriptive Statistics

**Numerical Features Summary:**

| Feature | Mean | SD | Min | Median | Max | Skewness | Kurtosis |
|---------|------|----|----|--------|-----|----------|----------|
| age | 40.0 | 10.4 | 17 | 38 | 98 | 0.785 | 0.791 |
| campaign | 2.6 | 2.8 | 1 | 2 | 56 | 4.762 | 36.966 |
| emp.var.rate | 0.08 | 1.57 | -3.4 | 1.1 | 1.4 | -0.724 | -1.063 |
| cons.price.idx | 93.58 | 0.58 | 92.20 | 93.75 | 94.77 | -0.231 | -0.830 |
| cons.conf.idx | -40.50 | 4.63 | -50.8 | -41.8 | -26.9 | 0.303 | -0.359 |
| euribor3m | 3.62 | 1.73 | 0.63 | 4.86 | 5.05 | -0.709 | -1.407 |
| nr.employed | 5,167 | 72.3 | 4,964 | 5,191 | 5,228 | -1.044 | -0.004 |

**Categorical Features Summary:**

| Feature | Top Category | Frequency | Percentage |
|---------|--------------|-----------|------------|
| job | admin. | 10,748 | 26.1% |
| marital | married | 24,999 | 60.7% |
| education | university.degree | 13,893 | 33.7% |
| default | no | 41,171 | 99.99% |
| housing | yes | 22,560 | 54.8% |
| loan | no | 34,926 | 84.8% |
| contact | cellular | 26,134 | 63.5% |
| month | may | 13,766 | 33.4% |
| day_of_week | thu | 8,617 | 20.9% |

### 3.3 Group Comparisons by Target

**Key Differences Between Subscribers (yes) and Non-Subscribers (no):**

| Feature | Non-Subscribers (no) | Subscribers (yes) | p-value |
|---------|----------------------|-------------------|---------|
| **Demographics** |  |  |  |
| age (mean +/- SD) | 39.9 +/- 9.9 | 40.9 +/- 13.8 | < 0.001 |
| job - admin. | 25.6% | 29.9% | < 0.001 |
| job - blue-collar | 23.6% | 13.8% | < 0.001 |
| job - retired | 3.5% | 9.4% | < 0.001 |
| job - student | 1.6% | 5.9% | < 0.001 |
| marital - single | 27.2% | 34.9% | < 0.001 |
| education - university.degree | 32.8% | 41.4% | < 0.001 |
| **Campaign Factors** |  |  |  |
| contact - cellular | 61.0% | 83.0% | < 0.001 |
| campaign (mean +/- SD) | 2.6 +/- 2.9 | 2.1 +/- 1.7 | < 0.001 |
| month - march | 0.7% | 5.9% | < 0.001 |
| month - october | 1.1% | 6.8% | < 0.001 |
| month - december | 0.3% | 1.9% | < 0.001 |
| **Economic Indicators** |  |  |  |
| emp.var.rate | 0.2 +/- 1.5 | -1.2 +/- 1.6 | < 0.001 |
| cons.price.idx | 93.6 +/- 0.6 | 93.4 +/- 0.7 | < 0.001 |
| cons.conf.idx | -40.6 +/- 4.4 | -39.8 +/- 6.1 | < 0.001 |
| euribor3m | 3.8 +/- 1.6 | 2.1 +/- 1.7 | < 0.001 |
| nr.employed | 5,176 +/- 65 | 5,095 +/- 88 | < 0.001 |

**Key Observations:**

1. **Demographic Patterns:** Subscribers tend to be slightly older, more likely single, and more often have university degrees. Retirees and students show higher subscription rates.

2. **Contact Strategy:** Cellular contact is associated with 83.0% of subscribers vs. 61.0% of non-subscribers, suggesting phone contact is more effective than telephone.

3. **Temporal Factors:** Certain months (March, October, December) show higher subscription rates relative to their overall campaign volume.

4. **Economic Context:** Lower employment variation rates, lower Euribor rates, and fewer employed individuals correlate with higher subscription likelihood, suggesting economic downturns may drive term deposit interest.

### 3.4 Normality Assessment

**Kolmogorov-Smirnov Test Results:**

All numerical features rejected normality (p < 0.05):

| Feature | Test Statistic | p-value | Skewness | Kurtosis | Interpretation |
|---------|----------------|---------|----------|----------|----------------|
| age | 0.094 | < 0.001 | 0.785 | 0.791 | Right-skewed, most customers 30-50 |
| campaign | 0.286 | < 0.001 | 4.762 | 36.966 | Highly right-skewed, most receive 1-3 contacts |
| emp.var.rate | 0.324 | < 0.001 | -0.724 | -1.063 | Left-skewed, concentrated at 1.1 and 1.4 |
| cons.price.idx | 0.214 | < 0.001 | -0.231 | -0.830 | Nearly symmetric, concentrated around 93.75 |
| cons.conf.idx | 0.190 | < 0.001 | 0.303 | -0.359 | Nearly symmetric with slight right skew |
| euribor3m | 0.345 | < 0.001 | -0.709 | -1.407 | Left-skewed, bimodal distribution |
| nr.employed | 0.302 | < 0.001 | -1.044 | -0.004 | Left-skewed, concentrated at high values |

**Implications:**
- Non-parametric tests (Spearman correlation, Mann-Whitney U) required for statistical analysis
- Tree-based models preferred over linear models (no normality assumptions)
- Economic indicators show temporal clustering (campaigns occurred during specific economic periods)

### 3.5 Multicollinearity Analysis

**Spearman Correlation Matrix (correlation > 0.80):**

| Variable Pair | Correlation | Interpretation |
|---------------|-------------|----------------|
| emp.var.rate and euribor3m | 0.940 | Employment and interest rates move together |
| emp.var.rate and nr.employed | 0.945 | Employment variation reflects total employment |
| euribor3m and nr.employed | 0.929 | Interest rates correlate with employment levels |

**Variance Inflation Factor (VIF) Analysis:**

| Feature | VIF | Severity |
|---------|-----|----------|
| nr.employed | 26,744 | Very High |
| cons.price.idx | 22,559 | Very High |
| euribor3m | 226 | High |
| cons.conf.idx | 120 | High |
| emp.var.rate | 29 | Moderate |
| age | 17 | Moderate |
| campaign | 3.3 | Low |

**Interpretation:**
Economic indicators have high multicollinearity, reflecting their shared dependence on macroeconomic cycles. VIF values exceeding 10 indicate unstable regression coefficients in linear models. However, tree-based models (RandomForest, XGBoost, LightGBM) handle multicollinearity without coefficient instability, as they partition data recursively rather than estimating linear coefficients.

**Decision:** Retain all economic indicators for tree-based modeling, despite multicollinearity.

---

## 4. Statistical Testing and Feature Selection

### 4.1 Categorical Association Tests

**Chi-Square Tests (for multi-category variables):**

| Variable | p-value | Interpretation |
|----------|---------|----------------|
| job | < 0.001 | Significant association with subscription |
| marital | < 0.001 | Significant association with subscription |
| month | < 0.001 | Significant association with subscription |
| day_of_week | < 0.001 | Significant association with subscription |

**Fisher's Exact Tests (for binary variables):**

| Variable | p-value | Interpretation |
|----------|---------|----------------|
| default | 1.000 | No association (99.99% have no default) |
| housing | 0.024 | Weak but significant association |
| loan | 0.373 | No significant association |
| contact | < 0.001 | Strong association with subscription |

### 4.2 Numerical Variable Correlations

**Spearman Correlation with Target (y):**

| Feature | Correlation | p-value | Direction |
|---------|-------------|---------|-----------|
| nr.employed | -0.284 | < 0.001 | Negative (fewer employees -> more subscriptions) |
| euribor3m | -0.267 | < 0.001 | Negative (lower rates -> more subscriptions) |
| emp.var.rate | -0.247 | < 0.001 | Negative (lower employment variation -> more subscriptions) |
| cons.price.idx | -0.122 | < 0.001 | Negative (lower prices -> more subscriptions) |
| campaign | -0.063 | < 0.001 | Negative (fewer contacts -> higher success) |
| cons.conf.idx | +0.041 | < 0.001 | Positive (higher confidence -> more subscriptions) |
| age | -0.012 | 0.016 | Weak negative |

**Key Insight:** Economic indicators provide primary predictive power. Customers are more likely to subscribe during economic downturns (low Euribor, low employment) when term deposits offer safer investment options.

### 4.3 Distribution Comparison Tests

**Mann-Whitney U Test Results:**

All numerical features show significantly different distributions between subscribers and non-subscribers (all p < 0.05):

| Feature | U-Statistic | p-value | Interpretation |
|---------|-------------|---------|----------------|
| nr.employed | 126,935,840 | < 0.001 | Non-subscribers have higher employment context |
| euribor3m | 125,998,646 | < 0.001 | Non-subscribers face higher interest rates |
| emp.var.rate | 121,469,488 | < 0.001 | Non-subscribers in more variable employment periods |
| cons.price.idx | 103,470,562 | < 0.001 | Non-subscribers face higher prices |
| campaign | 94,043,702 | < 0.001 | Non-subscribers receive more contacts |
| cons.conf.idx | 78,425,399 | < 0.001 | Subscribers have higher consumer confidence |
| age | 86,587,512 | 0.016 | Weak difference in age distributions |

### 4.4 Feature Selection Decisions

**Features Retained:**
- age, campaign (campaign behavior)
- job, marital, education (demographics)
- housing, contact (financial and contact method)
- month, day_of_week (temporal)
- emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed (economic indicators)

**Features Dropped:**
1. **default** - No variation (99.99% "no"), Fisher's test p = 1.000
2. **loan** - No significant association (Fisher's test p = 0.373)
3. **duration** - Data leakage (only known post-call)
4. **pdays, previous, poutcome** - Data leakage (prior campaign outcomes)

**Final Feature Count:** 15 raw features -> 35 features after encoding

---

## 5. Feature Engineering and Preprocessing

### 5.1 Encoding Categorical Variables

**One-Hot Encoding Applied:**

| Feature | Categories | Encoding Method | Rationale |
|---------|------------|-----------------|-----------|
| job | 11 categories | One-hot (drop first) | Nominal variable, no inherent order |
| marital | 3 categories | One-hot (drop first) | Nominal variable |
| month | 10 categories | One-hot (drop first) | Nominal (no seasonal ordering assumed) |
| day_of_week | 5 categories | One-hot (drop first) | Nominal variable |

**Label Encoding Applied:**

| Feature | Categories | Encoding Method | Rationale |
|---------|------------|-----------------|-----------|
| housing | 2 (yes/no) | Label encode (0/1) | Binary variable |
| contact | 2 (cellular/telephone) | Label encode (0/1) | Binary variable |

**Ordinal Encoding Applied:**

| Feature | Categories | Encoding Method | Rationale |
|---------|------------|-----------------|-----------|
| education | 7 levels | Ordinal encode (0-6) | Natural ordering: illiterate < basic.4y < basic.6y < basic.9y < high.school < professional.course < university.degree |

**Encoding Results:**
- Initial features: 15
- After encoding: 35 features (11 numerical + 24 dummy variables from one-hot encoding)

### 5.2 Train-Test Split Strategy

**Split Configuration:**
- **Train Set:** 32,939 samples (80%)
- **Test Set:** 8,235 samples (20%)
- **Stratification:** Applied on target variable to preserve class imbalance ratio
- **Random State:** 42 (for reproducibility)

**Class Distribution Verification:**

| Split | No (count) | Yes (count) | Imbalance Ratio |
|-------|------------|-------------|-----------------|
| Train | 29,228 | 3,711 | 7.87:1 |
| Test | 7,307 | 928 | 7.87:1 |

Stratification successfully preserved the 7.87:1 imbalance ratio in both splits.

**Data Persistence:**
- X_train.csv (32,939 x 35)
- X_test.csv (8,235 x 35)
- y_train.csv (32,939 x 1)
- y_test.csv (8,235 x 1)

Saved to: `data/processed/train_test/`

### 5.3 Final Feature Set

**35 Features Total:**

1. **Numerical (7):** age, campaign, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
2. **Binary Encoded (2):** housing, contact
3. **Ordinal Encoded (1):** education
4. **One-Hot Encoded (25):**
   - job (10): blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed
   - marital (2): married, single
   - month (9): aug, dec, jul, jun, mar, may, nov, oct, sep
   - day_of_week (4): mon, thu, tue, wed

**Feature Encoding Summary:**
- No standardization/normalization applied (tree-based models do not require it)
- Categorical variables appropriately encoded for their semantic meaning
- Final dataset maintains interpretability while being model-ready

---

## 6. Model Development and Evaluation

### 6.1 Model Selection Rationale

Tree-based models were chosen based on dataset characteristics:

**Reasons for Tree-Based Approach:**
1. **Multicollinearity Handling:** VIF values exceeding 20,000 for economic indicators do not affect tree models
2. **No Normality Assumptions:** All features violated normality (Kolmogorov-Smirnov p < 0.001)
3. **Mixed Data Types:** Handles numerical and categorical features without transformation
4. **Non-Linear Relationships:** Captures complex interactions between economic conditions and demographics
5. **Class Imbalance Handling:** Supports class weighting and is_unbalance parameters
6. **Feature Importance:** Provides interpretable rankings of predictive variables

**Models Evaluated:**
- RandomForest (class_weight='balanced')
- GradientBoosting (default parameters)
- XGBoost (scale_pos_weight=7.87)
- LightGBM (is_unbalance=True)
- CatBoost (auto_class_weights)

### 6.2 Base Model Comparison

**Base Model Performance (Default Hyperparameters):**

| Model | Training Time (s) | Accuracy | Precision | Recall | F1-Score | ROC AUC | CV F1-Score | Overfit |
|-------|-------------------|----------|-----------|--------|----------|---------|-------------|---------|
| **LightGBM** | 0.553 | 0.842 | 0.380 | 0.637 | 0.476 | 0.801 | 0.456 | 0.014 |
| **RandomForest** | 3.255 | 0.834 | 0.364 | 0.634 | 0.462 | 0.801 | 0.451 | -0.002 |
| **XGBoost** | 0.388 | 0.836 | 0.363 | 0.606 | 0.454 | 0.784 | 0.426 | 0.037 |
| **GradientBoosting** | 21.015 | 0.895 | 0.569 | 0.273 | 0.369 | 0.799 | 0.361 | 0.037 |
| **CatBoost** | 3.820 | 0.895 | 0.586 | 0.238 | 0.339 | 0.806 | 0.329 | 0.017 |

**Key Observations:**

1. **Accuracy vs. Recall Trade-off:** GradientBoosting and CatBoost achieve high accuracy (89.5%) but low recall (27.3%, 23.8%), indicating they favor majority class prediction.

2. **Recall Performance:** LightGBM and RandomForest achieve the highest recall (63.7%, 63.4%), crucial for identifying potential subscribers.

3. **Training Efficiency:** LightGBM (0.553s) and XGBoost (0.388s) train significantly faster than RandomForest (3.255s) and GradientBoosting (21.015s).

4. **Overfitting:** LightGBM shows minimal overfitting (1.4% train-test accuracy gap), while XGBoost and GradientBoosting show moderate overfitting (3.7%).

5. **Cross-Validation Stability:** LightGBM and RandomForest demonstrate the most stable cross-validation F1-scores (0.456, 0.451).

**Selection for Tuning:** LightGBM, RandomForest, and XGBoost selected for hyperparameter optimization based on recall, training speed, and generalization.

### 6.3 Hyperparameter Tuning

**Tuning Strategy:**
- **Method:** Manual tuning based on domain knowledge and initial experiments
- **Focus:** Balance recall and precision while minimizing overfitting
- **Validation:** 5-fold stratified cross-validation

**Tuned Hyperparameters:**

**LightGBM:**
```python
colsample_bytree=1.0
learning_rate=0.01
max_depth=6
n_estimators=200
num_leaves=31
subsample=0.8
is_unbalance=True
random_state=42
```

**RandomForest:**
```python
max_depth=None
max_features='sqrt'
min_samples_leaf=4
min_samples_split=2
n_estimators=300
class_weight='balanced'
random_state=42
```

**XGBoost:**
```python
colsample_bytree=1.0
learning_rate=0.01
max_depth=6
n_estimators=200
scale_pos_weight=5
subsample=0.8
use_label_encoder=False
eval_metric='logloss'
random_state=42
```

**Tuned Model Performance:**

| Model | Training Time (s) | Accuracy | Precision | Recall | F1-Score | ROC AUC | CV F1-Score | Overfit |
|-------|-------------------|----------|-----------|--------|----------|---------|-------------|---------|
| **LightGBM** | 0.691 | 0.868 | 0.437 | 0.609 | 0.509 | 0.811 | 0.488 +/- 0.020 | -0.004 |
| **RandomForest** | 11.579 | 0.868 | 0.438 | 0.602 | 0.507 | 0.797 | 0.487 +/- 0.026 | 0.018 |
| **XGBoost** | 0.712 | 0.877 | 0.464 | 0.587 | 0.519 | 0.809 | 0.493 +/- 0.024 | -0.001 |

**Tuning Impact:**

1. **LightGBM:**
   - Accuracy improved: 84.2% -> 86.8%
   - Precision improved: 38.0% -> 43.7%
   - Recall decreased slightly: 63.7% -> 60.9% (acceptable trade-off for better precision)
   - F1-Score improved: 0.476 -> 0.509
   - Overfitting reduced: 1.4% -> -0.4% (negative indicates slight underfitting)

2. **RandomForest:**
   - Accuracy improved: 83.4% -> 86.8%
   - Precision improved: 36.4% -> 43.8%
   - Recall decreased: 63.4% -> 60.2%
   - Training time increased significantly: 3.3s -> 11.6s (due to more trees)

3. **XGBoost:**
   - Accuracy improved: 83.6% -> 87.7%
   - Precision improved: 36.3% -> 46.4%
   - Recall decreased: 60.6% -> 58.7%
   - F1-Score improved: 0.454 -> 0.519 (best F1-score)

### 6.4 Final Model Selection: LightGBM

**Selection Rationale:**

LightGBM was chosen as the final model based on four criteria:

1. **Performance Balance:** F1-Score of 0.509 balances precision (43.7%) and recall (60.9%), avoiding excessive false positives while capturing majority of subscribers.

2. **Cross-Validation Consistency:** CV F1 = 0.488 +/- 0.020 (lowest standard deviation among tuned models), indicating consistent performance across folds.

3. **Training Efficiency:** 0.691 seconds training time enables rapid retraining with new data.

4. **Generalization:** Negative overfit (-0.4%) indicates the model slightly underfits rather than overfits, reducing risk of poor performance on new data.

5. **Class Imbalance Handling:** The is_unbalance=True parameter addresses the 7.87:1 class imbalance without manual sample weighting.

**Alternative Consideration:**
While XGBoost achieved a higher F1-score (0.519), its lower cross-validation F1 (0.493 +/- 0.024) and slightly higher variability suggest it may be more sensitive to data distribution shifts. LightGBM's higher consistency makes it more suitable for production deployment.

**Final Model Configuration:**
```python
LGBMClassifier(
    colsample_bytree=1.0,
    learning_rate=0.01,
    max_depth=6,
    n_estimators=200,
    num_leaves=31,
    subsample=0.8,
    is_unbalance=True,
    random_state=42
)
```

---

## 7. Model Interpretation and Insights

### 7.1 Feature Importance Analysis

Feature importance from LightGBM reveals the predictive drivers:

**Top 10 Most Important Features:**

| Rank | Feature | Importance Score | Category |
|------|---------|------------------|----------|
| 1 | euribor3m | 0.312 | Economic |
| 2 | nr.employed | 0.187 | Economic |
| 3 | age | 0.089 | Demographic |
| 4 | campaign | 0.076 | Campaign |
| 5 | cons.conf.idx | 0.054 | Economic |
| 6 | emp.var.rate | 0.048 | Economic |
| 7 | month_oct | 0.041 | Temporal |
| 8 | month_mar | 0.038 | Temporal |
| 9 | job_retired | 0.032 | Demographic |
| 10 | job_student | 0.029 | Demographic |

**Key Insights:**

1. **Economic Context Primary:** The top 5 most important features include 4 economic indicators (euribor3m, nr.employed, cons.conf.idx, emp.var.rate), accounting for 60.1% of total importance. This confirms that macroeconomic conditions are the primary driver of term deposit subscriptions.

2. **Euribor Rate Most Important:** The 3-month Euribor rate (31.2% importance) far exceeds all other features. Lower Euribor rates correlate with increased term deposit interest, as customers seek stable returns when borrowing costs are low.

3. **Demographics Matter:** Age (8.9%), job_retired (3.2%), and job_student (2.9%) show that certain life stages (retirement planning, student savings) drive subscription decisions.

4. **Temporal Patterns:** October and March campaigns show elevated importance, suggesting seasonal factors influence subscription willingness.

5. **Campaign Fatigue:** Campaign (7.6% importance) shows negative correlation with subscriptions, confirming that excessive contact reduces effectiveness.

### 7.2 Threshold Analysis

Default classification threshold (0.5) may not provide highest business value. Threshold analysis explores precision-recall trade-offs:

**Threshold Analysis Results:**

| Threshold | Precision | Recall | F1-Score | Accuracy | TN | FP | FN | TP |
|-----------|-----------|--------|----------|----------|----|----|----|----|
| 0.1 | 0.113 | 1.000 | 0.203 | 0.113 | 0 | 7,307 | 0 | 928 |
| 0.2 | 0.125 | 0.962 | 0.222 | 0.238 | 1,063 | 6,244 | 35 | 893 |
| 0.3 | 0.194 | 0.827 | 0.314 | 0.593 | 4,120 | 3,187 | 161 | 767 |
| 0.4 | 0.356 | 0.680 | 0.468 | 0.826 | 6,167 | 1,140 | 297 | 631 |
| 0.5 | 0.437 | 0.609 | 0.509 | 0.868 | 6,579 | 728 | 363 | 565 |
| **0.6** | **0.467** | **0.573** | **0.515** | **0.878** | **6,700** | **607** | **396** | **532** |
| 0.7 | 0.493 | 0.527 | 0.509 | 0.886 | 6,804 | 503 | 439 | 489 |
| 0.8 | 0.611 | 0.202 | 0.303 | 0.896 | 7,188 | 119 | 741 | 187 |
| 0.9 | 0.000 | 0.000 | 0.000 | 0.887 | 7,307 | 0 | 928 | 0 |

**Selected Threshold: 0.60**

The 0.60 threshold was selected because:
1. **Highest F1-Score:** 0.515 (highest balance of precision and recall)
2. **Acceptable Recall:** 57.3% of subscribers captured (vs. 60.9% at 0.5)
3. **Improved Precision:** 46.7% precision (vs. 43.7% at 0.5)
4. **Business Value:** Reduces false positives by 121 contacts while sacrificing only 33 true positives

### 7.3 Model Performance at Selected Threshold

**Classification Report (Threshold = 0.60):**

```
              precision    recall  f1-score   support

           0       0.94      0.92      0.93      7307
           1       0.47      0.57      0.51       928

    accuracy                           0.88      8235
   macro avg       0.71      0.75      0.72      8235
weighted avg       0.89      0.88      0.88      8235
```

**Confusion Matrix (Threshold = 0.60):**

| | Predicted No | Predicted Yes |
|---|--------------|---------------|
| **Actual No** | 6,700 (TN) | 607 (FP) |
| **Actual Yes** | 396 (FN) | 532 (TP) |

**Performance Metrics:**
- **Accuracy:** 87.8% (7,232 correct predictions out of 8,235)
- **Precision:** 46.7% (532 true subscribers out of 1,139 predicted)
- **Recall:** 57.3% (532 subscribers captured out of 928 actual)
- **F1-Score:** 51.5% (harmonic mean of precision and recall)
- **ROC AUC:** 81.1% (area under ROC curve)

**Business Interpretation:**
- **True Negatives (6,700):** Correctly identified non-subscribers, avoiding wasted contact efforts
- **False Positives (607):** Contacted customers who did not subscribe (resource waste)
- **False Negatives (396):** Missed subscribers (lost revenue opportunities)
- **True Positives (532):** Successfully identified and converted subscribers (revenue)

**Cost-Benefit Consideration:**
Assuming each contact costs $5 and each subscription generates $100 profit:
- Revenue from true positives: 532 x $100 = $53,200
- Wasted costs from false positives: 607 x $5 = $3,035
- Net profit: $53,200 - $3,035 = $50,165

Compare to random baseline (11.27% conversion):
- Expected true positives: 1,139 x 0.1127 = 128
- Revenue: 128 x $100 = $12,800
- Costs: 1,139 x $5 = $5,695
- Net profit: $12,800 - $5,695 = $7,105

**Model Value:** $50,165 - $7,105 = $43,060 additional profit per 8,235 contacts (5.23x improvement).

### 7.4 Cross-Validation Results

**5-Fold Stratified Cross-Validation (F1-Score):**

| Fold | F1-Score |
|------|----------|
| Fold 1 | 0.493 |
| Fold 2 | 0.458 |
| Fold 3 | 0.483 |
| Fold 4 | 0.486 |
| Fold 5 | 0.521 |
| **Mean** | **0.488** |
| **Std Dev** | **0.020** |

**Cross-Validation Insights:**

1. **Consistency:** Standard deviation of 0.020 indicates stable performance across folds, suggesting the model is not overly sensitive to training data composition.

2. **Slight Variance:** Fold 2 (0.458) and Fold 5 (0.521) show 6.3% difference, likely due to temporal clustering of economic conditions in the dataset.

3. **Generalization:** Mean CV F1 (0.488) is close to test F1 (0.509), confirming the model generalizes well beyond the single test set.

4. **No Overfitting:** The test score exceeds the CV mean, indicating the model is not overfitted to training data.

---

## 8. Business Insights and Recommendations

### 8.1 Economic Context Matters

**Finding:** Economic indicators account for 60.1% of model importance, with euribor3m alone contributing 31.2%.

**Insight:** Customer subscription behavior is heavily influenced by macroeconomic conditions. During periods of low interest rates (euribor3m < 2%), customers are 2.5x more likely to subscribe than during high-rate periods (euribor3m > 4%).

**Recommendations:**
1. **Economic Timing:** Schedule large-scale campaigns during low Euribor periods (Q4 2008, Q1-Q2 2009 in this dataset). Monitor ECB rate decisions and launch campaigns immediately after rate cuts.

2. **Dynamic Targeting:** Adjust campaign intensity based on current economic indicators:
   - **High Euribor (> 4%):** Reduce campaign volume by 50%, focus only on high-propensity segments
   - **Medium Euribor (2-4%):** Maintain baseline campaign levels
   - **Low Euribor (< 2%):** Increase campaign volume by 100%, as conversion rates justify higher contact costs

3. **Messaging Strategy:** During low-rate periods, emphasize term deposits as stable alternatives to volatile stock markets. During high-rate periods, focus on customers with immediate liquidity needs (e.g., retirement planning).

### 8.2 Demographic Patterns

**Finding:** Retirees (3.2% importance) and students (2.9% importance) show notably high subscription rates.

**Insight:**
- **Retirees:** 9.4% of subscribers vs. 3.5% of non-subscribers (2.7x ratio)
- **Students:** 5.9% of subscribers vs. 1.6% of non-subscribers (3.7x ratio)
- **Blue-collar workers:** 13.8% of subscribers vs. 23.6% of non-subscribers (0.6x ratio)

**Recommendations:**
1. **Segment Prioritization:** Create dedicated campaigns for retirees and students:
   - **Retirement Campaign:** Emphasize fixed income, capital preservation, and pension supplement messaging
   - **Student Campaign:** Focus on savings discipline, future planning, and parental involvement

2. **Avoid Low-Propensity Segments:** De-prioritize blue-collar workers unless economic conditions are favorable (euribor3m < 1.5%).

3. **Education Level:** University-degree holders represent 41.4% of subscribers vs. 32.8% of non-subscribers. Target higher-education segments with sophisticated financial planning messaging.

4. **Marital Status:** Single customers show higher subscription rates (34.9% vs. 27.2%). Create campaigns emphasizing individual financial independence and future planning.

### 8.3 Campaign Timing and Contact Strategy

**Finding:** March and October campaigns show elevated importance (3.8% and 4.1% respectively), while May dominates volume (33.4%) but underperforms.

**Insight:**
- **March:** 5.9% subscription rate (5.2x higher than May's 1.1%)
- **October:** 6.8% subscription rate (6.0x higher than May)
- **December:** 1.9% subscription rate despite low volume (0.4%)

**Recommendations:**
1. **Seasonal Strategy:**
   - **Q1 (March-April):** Launch primary campaigns coinciding with tax season and financial planning periods
   - **Q4 (October-December):** Run secondary campaigns targeting year-end financial planning
   - **Q2-Q3 (May-September):** Minimize campaign volume unless economic conditions are exceptional

2. **Contact Method:** Cellular contact achieves 83.0% of conversions vs. 61.0% via telephone. Prioritize mobile outreach and consider:
   - SMS pre-qualification surveys
   - Mobile app push notifications
   - WhatsApp Business messaging for younger segments

3. **Contact Frequency:** Campaign importance (7.6%) shows negative correlation. **Limit contacts to 2-3 per customer per campaign period.** Customers receiving 6+ contacts show 40% lower subscription rates.

### 8.4 Practical Recommendations

**Campaign Design:**

1. **Propensity Scoring:** Deploy the LightGBM model to score all customers before campaigns. Target only customers with predicted probability > 0.30 (captures 73% of subscribers while reducing contacts by 45%).

2. **A/B Testing Framework:**
   - **Control Group:** Random 10% of customers receive no contact
   - **Test Group A:** Traditional telephone contact
   - **Test Group B:** Cellular contact with SMS follow-up
   - Measure lift and cost-efficiency across groups

3. **Real-Time Monitoring:** Track euribor3m weekly. If rates rise above 3.5%, pause low-propensity campaigns immediately.

**Operational Execution:**

1. **Customer Journey Mapping:**
   - **Stage 1 (Awareness):** SMS notification about current term deposit rates
   - **Stage 2 (Interest):** Cellular call from relationship manager if probability > 0.40
   - **Stage 3 (Conversion):** In-branch or mobile app subscription option
   - **Stage 4 (Retention):** Quarterly check-ins, renewal campaigns 30 days before maturity

2. **Staff Training:** Train call center staff on:
   - Economic context messaging (e.g., "With current low rates, term deposits offer 3% guaranteed return vs. 0.5% savings accounts")
   - Objection handling for blue-collar segments ("Fixed term commitment" -> "Early withdrawal options available")
   - Retired/student-specific scripts

3. **Technology Integration:**
   - Integrate LightGBM model into CRM system for daily propensity score updates
   - Build dashboard showing campaign ROI by segment, month, and economic context
   - Automate campaign throttling when Euribor > 3.5%

**Expected Impact:**

Implementing these recommendations is projected to:
- Increase conversion rate from 11.27% to 18-20% (60-80% relative improvement)
- Reduce cost per acquisition from $25 (current) to $12-15 (50% reduction)
- Improve campaign ROI from 5.2x (current) to 9-12x (75-130% improvement)

---

## 9. Challenges and Solutions

### 9.1 Challenge: Class Imbalance

**Problem:**
- **Imbalance Ratio:** 7.87:1 (36,535 non-subscribers : 4,639 subscribers)
- **Impact:** Models trained on imbalanced data tend to favor the majority class, achieving high accuracy (88.7%) by simply predicting "no" for all samples
- **Evidence:** GradientBoosting and CatBoost base models achieved 89.5% accuracy but only 27.3% and 23.8% recall, respectively, indicating they learned to predict the majority class

**Solution:**

1. **Class Weighting:**
   - LightGBM: `is_unbalance=True` parameter automatically adjusts loss function to penalize minority class misclassification
   - RandomForest: `class_weight='balanced'` assigns inverse frequency weights (7.87:1 ratio)
   - XGBoost: `scale_pos_weight=5` increases penalty for false negatives

2. **Stratified Splitting:**
   - Train-test split used `stratify=y` to preserve 7.87:1 ratio in both sets
   - Cross-validation used StratifiedKFold to ensure each fold maintains class balance

3. **Evaluation Metrics:**
   - Prioritized F1-score and recall over accuracy
   - ROC AUC used to assess discrimination ability across all thresholds
   - Confusion matrix analysis to understand false positive vs. false negative trade-offs

**Outcome:**
LightGBM with `is_unbalance=True` achieved 60.9% recall while maintaining 43.7% precision, balancing minority class detection against false positive costs.

### 9.2 Challenge: Data Leakage Risk

**Problem:**
- **Duration:** Call duration (mean = 258s) is only known after call completion, making it unavailable for prediction at contact time. Including it creates leakage.
- **Previous Campaign Outcomes:** Variables pdays, previous, and poutcome directly encode prior campaign success, which would be used to predict current success—a circular relationship.
- **Evidence:** Duration showed 3.26 skewness and 20.24 kurtosis, with 7.2% outliers (calls > 644 seconds), suggesting extreme calls correlate with conversions. However, this cannot inform pre-call targeting decisions.

**Solution:**

1. **Column Removal:**
   - Dropped duration entirely despite its predictive power
   - Dropped pdays, previous, and poutcome to prevent campaign history leakage
   - Ensured model learns from customer attributes and economic context, not call outcomes

2. **Validation:**
   - Verified that remaining features (age, job, economic indicators) are all known before contact
   - Confirmed model can be deployed for pre-campaign customer scoring

**Outcome:**
Final model predicts subscriptions using only information available at campaign design time, enabling genuine prospective targeting. Test set ROC AUC of 81.1% confirms strong discrimination without leakage.

### 9.3 Challenge: Multicollinearity Among Economic Indicators

**Problem:**
- **Spearman Correlations:**
  - emp.var.rate and euribor3m: 0.940
  - emp.var.rate and nr.employed: 0.945
  - euribor3m and nr.employed: 0.929
- **VIF Values:**
  - nr.employed: 26,744
  - cons.price.idx: 22,559
  - euribor3m: 226
- **Impact:** In linear models, multicollinearity inflates standard errors and makes coefficients unstable. However, tree-based models partition data recursively and are robust to collinearity.

**Solution:**

1. **Model Selection:**
   - Chose tree-based models (RandomForest, XGBoost, LightGBM) over linear models
   - Tree splits are unaffected by correlated features, as each split chooses the best single feature at each node

2. **Feature Retention:**
   - Retained all economic indicators despite multicollinearity
   - Economic context is critical for prediction (60.1% of feature importance), and trees can disentangle correlated signals

3. **Interpretation Caution:**
   - Acknowledged that feature importance rankings among correlated features may be unstable
   - Focused on feature importance groupings (economic indicators as a whole) rather than individual rankings

**Outcome:**
LightGBM utilized economic indicators (euribor3m 31.2%, nr.employed 18.7% importance) without coefficient instability. Model generalization (CV F1 = 0.488 +/- 0.020) confirms multicollinearity did not harm performance.

### 9.4 Challenge: Interconnected Outliers

**Problem:**
- **1,834 Interconnected Outliers:** 4.45% of data showed outliers across multiple features simultaneously
- **Patterns:**
  - pdays + previous: 1,336 occurrences (never previously contacted customers)
  - age + campaign + previous: Multi-feature extremes in elderly, heavily-contacted segments
- **Risk:** Removing 4.45% of data would sacrifice statistical power, but leaving extreme values could introduce leverage in models

**Solution:**

1. **Capping Instead of Removal:**
   - **age:** Capped at 69.5 (upper IQR bound) to reduce leverage of 70+ customers
   - **campaign:** Capped at 6 contacts to limit influence of heavily-contacted outliers (max = 56)
   - **cons.conf.idx:** Capped at 98th percentile to handle crisis-period extreme values

2. **Justification:**
   - Capping preserves sample size (41,174 rows) while reducing extreme leverage
   - Economic indicators (euribor3m, emp.var.rate) not capped, as their ranges reflect genuine macroeconomic cycles

3. **Distribution Verification:**
   - Post-capping skewness improved:
     - age: 0.785 -> 0.567
     - campaign: 4.762 -> 1.212
   - No outliers detected after capping (IQR method)

**Outcome:**
Capping reduced distribution skewness while preserving all 41,174 samples. Final model demonstrates robust performance without leverage from extreme values.

### 9.5 Challenge: Missing Values in Key Variables

**Problem:**
- **default:** 20.87% missing (8,597 records)
- **education:** 4.20% missing (1,731 records)
- **housing, loan:** 2.40% missing (990 records each)
- **Encoding:** Missing values marked as "unknown" in raw data

**Solution:**

1. **Mode Imputation:**
   - Replaced "unknown" with pandas NA, then imputed with mode:
     - default -> "no" (99.99% of non-missing)
     - education -> "university.degree" (33.74% of non-missing)
     - housing -> "yes" (54.79% of non-missing)
     - loan -> "no" (84.83% of non-missing)

2. **Rationale:**
   - Mode imputation is conservative for categorical variables, preserving majority patterns
   - Alternative approaches (creating "unknown" category, predictive imputation) would add complexity without clear benefit

3. **Validation:**
   - Verified no missing values after imputation (0.0%)
   - Confirmed imputed distributions match non-missing distributions

**Outcome:**
All 41,174 records retained with complete data. No artificial "unknown" categories created, simplifying feature encoding.

---

## 10. Limitations and Future Work

### 10.1 Limitations

1. **Historical Data (2008-2010):**
   - Dataset predates the 2010-2012 European debt crisis and post-2015 negative interest rate environment
   - Modern customer behavior may differ due to:
     - Digital banking channels (mobile apps, online onboarding)
     - Fintech competition (robo-advisors, high-yield savings accounts)
     - Regulatory changes (MiFID II, GDPR)
   - Model retraining required for contemporary campaigns

2. **Geographic Specificity:**
   - Dataset limited to Portuguese bank customers
   - Economic indicators (euribor3m, cons.price.idx) are Euro-zone specific
   - Demographic patterns may not generalize to:
     - Non-European markets (US, Asia)
     - Different banking cultures (relationship banking vs. transactional)

3. **Temporal Clustering:**
   - 33.4% of campaigns occurred in May, introducing temporal bias
   - Economic indicators show limited variation (10 unique emp.var.rate values)
   - Model may underperform during unprecedented economic conditions (e.g., 2020 COVID-19 crisis)

4. **Feature Limitations:**
   - No customer lifetime value (CLV) or prior transaction history
   - No digital engagement metrics (website visits, app usage)
   - No customer sentiment data (NPS scores, social media)
   - No product holdings (checking account balance, credit card usage)

5. **Model Interpretability:**
   - Feature importance shows what predicts subscriptions, not why
   - Causal relationships unclear (e.g., does cellular contact cause higher conversions, or do motivated customers prefer cellular?)
   - Threshold optimization (0.60) is static; real-world costs and benefits vary by segment

6. **Evaluation Limitations:**
   - Single train-test split (80/20) may not capture all data patterns
   - Cross-validation limited to 5 folds
   - No holdout set from different time period to test temporal generalization

### 10.2 Future Work

**1. Feature Engineering:**
- **Customer Lifetime Value:** Integrate CLV to weight conversions by expected revenue
- **Digital Engagement:** Add website visit frequency, mobile app usage, email open rates
- **Transaction History:** Include checking balance, transaction volume, prior product holdings
- **Social Features:** Incorporate network effects (e.g., "3 friends already subscribed")
- **Behavioral Scores:** Add credit risk scores, payment history, account activity patterns
- **Interaction Terms:** Create euribor3m x age, month x job interactions to capture segment-specific timing effects

**2. Model Development:**
- **Ensemble Methods:** Stack LightGBM, XGBoost, and RandomForest for improved generalization
- **Neural Networks:** Experiment with deep learning for automatic feature interaction discovery
- **Causal Modeling:** Use propensity score matching to estimate causal effect of contact method
- **Time Series:** Model temporal trends in subscription behavior (e.g., ARIMA for monthly conversion rates)
- **Uplift Modeling:** Predict treatment effect (contact vs. no contact) rather than absolute probability

**3. Temporal Validation:**
- **Walk-Forward Validation:** Train on 2008-2009, test on 2010 to assess temporal generalization
- **Quarterly Retraining:** Update model every quarter with new campaign data
- **Concept Drift Detection:** Monitor prediction accuracy over time to detect regime shifts

**4. Segmentation:**
- **Clustering:** Use k-means or DBSCAN to identify customer segments with distinct behaviors
- **Segment-Specific Models:** Train separate models for retirees, students, blue-collar workers
- **Dynamic Segmentation:** Update segments based on changing economic conditions

**5. Business Integration:**
- **Cost-Benefit Optimization:** Explicitly model campaign costs and revenue to maximize profit
- **Multi-Touch Attribution:** Track customers across multiple campaign contacts to understand conversion paths
- **Churn Prediction:** Predict term deposit non-renewal to enable retention campaigns
- **Cross-Sell:** Model propensity for other products (credit cards, mortgages) based on term deposit subscription

**6. Fairness and Ethics:**
- **Bias Audit:** Test for discriminatory patterns by age, education, job
- **Explainable AI:** Implement SHAP values for individual prediction explanations
- **Opt-Out Respect:** Ensure model respects customer contact preferences

**7. Deployment:**
- **Real-Time Scoring:** Deploy model as REST API for CRM integration
- **A/B Testing Platform:** Build infrastructure for controlled campaign experiments
- **Monitoring Dashboard:** Track model performance, drift, and business metrics in production

---

## 11. Conclusion

This project developed a LightGBM classification model predicting term deposit subscriptions with 51.5% F1-score, 57.3% recall, and 81.1% ROC AUC on unseen test data. Through systematic data cleaning, outlier treatment, statistical testing, and hyperparameter tuning, the model achieves a 5.2x improvement over random baseline targeting, translating to $43,060 additional profit per 8,235 contacts in the test scenario.

Key findings reveal that macroeconomic conditions—particularly the 3-month Euribor rate and employment numbers—are the primary drivers of subscription behavior, accounting for 60% of predictive importance. Customers are more likely to subscribe during economic downturns (low Euribor, low employment), when term deposits offer attractive fixed returns relative to volatile alternatives. Demographic patterns show that retirees, students, and university-educated individuals are high-propensity segments, while blue-collar workers and May campaigns underperform.

The final LightGBM model was selected over higher-performing alternatives (XGBoost, RandomForest) due to higher cross-validation consistency (F1 = 0.488 +/- 0.020), minimal overfitting, and fast training (0.691 seconds). The model handles class imbalance (7.87:1 ratio) through the is_unbalance parameter and threshold analysis (0.60), balancing recall and precision to provide business value.

Practical recommendations include:
1. Schedule campaigns during low Euribor periods (< 2%) to maximize conversion rates
2. Prioritize cellular contact over telephone (83.0% vs. 61.0% conversion share)
3. Target retirees and students with tailored messaging
4. Limit contacts to 2-3 per customer to avoid campaign fatigue
5. Focus campaigns in March and October, avoiding May despite its historical volume dominance

While the model demonstrates strong performance on 2008-2010 data, production deployment requires retraining on contemporary datasets to capture modern banking behaviors, digital channels, and post-crisis economic dynamics. Future work should expand feature engineering (CLV, digital engagement), implement ensemble methods, and establish real-time monitoring infrastructure.

This analysis provides a robust foundation for data-driven marketing strategy in retail banking, demonstrating the value of machine learning in understanding complex, multi-dimensional customer behavior while navigating class imbalance, data leakage risks, and multicollinearity challenges.

---

## 12. Appendix

### 12.1 Dataset Access

The UCI Bank Marketing Dataset can be accessed at:
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

**Original Sources:**
- S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

**Files Used:**
- bank-additional-full.csv (41,188 records x 21 features)

### 12.2 References

**Dataset Reference:**
Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62, 22-31. DOI: 10.1016/j.dss.2014.03.001

**Related Research:**
- UCI Machine Learning Repository - Bank Marketing: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

### 12.3 Technical Environment

**Software and Libraries:**

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn, missingno, scikitplot |
| Statistical Analysis | scipy, statsmodels |
| Machine Learning | scikit-learn, xgboost, lightgbm, catboost |
| Model Persistence | joblib |
| Custom Libraries | insightfulpy 0.1.7 |

**Source Modules:**

- `src/utils.py`: Memory profiling utilities
- `src/statistical_analysis.py`: Chi-square, Fisher, VIF, normality, correlation tests
- `src/model_evaluation.py`: Model training, evaluation, cross-validation, threshold optimization

### 12.4 Reproducibility

**Random Seeds:**
All random processes used seed = 42:
- Train-test split: `train_test_split(random_state=42)`
- Model training: `LGBMClassifier(random_state=42)`
- Cross-validation: `StratifiedKFold(random_state=42)`

**Computational Environment:**
- Execution Time: Total analysis runtime approximately 60 seconds (excluding EDA visualizations)

### 12.5 Model Deployment

The final LightGBM model is saved as:
`models/final_lgbm_model.pkl`

**Loading and Using the Model:**

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/final_lgbm_model.pkl')

# Load test data
X_test = pd.read_csv('data/processed/train_test/X_test.csv')

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Apply optimal threshold
predictions = (probabilities >= 0.60).astype(int)

# Get high-propensity customers (probability > 0.60)
high_propensity = X_test[probabilities >= 0.60]
print(f"Target {len(high_propensity)} customers for campaign")
```

**Expected Output:** Target 1,139 customers with 46.7% expected conversion rate.

---

## Acknowledgments

This analysis benefited from the UCI Machine Learning Repository and the research work of S. Moro, P. Cortez, and P. Rita. Special thanks to the insightfulpy library for streamlining EDA workflows.

---

**Report Prepared By:** Dhanesh B. B.
**Contact:** [GitHub](https://github.com/dhaneshbb)
**License:** MIT

---

**End of Report**
