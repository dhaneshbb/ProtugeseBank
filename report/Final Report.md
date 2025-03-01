# Table of Contents

- [Data Analysis](#data-analysis)
    - [Import data](#import-data)
    - [Imports & functions](#imports-functions)
  - [Data understanding and cleaning](#data-understanding-and-cleaning)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Univariate Analysis](#univariate-analysis)
      - [num_analysis](#num-analysis)
      - [cat_analysis](#cat-analysis)
    - [Bivariate & Multivariate Analysis](#bivariate-multivariate-analysis)
- [Predictive Model](#predictive-model)
  - [Preprocessing](#preprocessing)
      - [outliers](#outliers)
      - [relation](#relation)
      - [Association](#association)
      - [Encoding](#encoding)
      - [Splitting](#splitting)
  - [Model Development](#model-development)
    - [Model Training & Evaluation](#model-training-evaluation)
    - [Model comparision & Interpretation](#model-comparision-interpretation)
    - [Best Model](#best-model)
- [Acknowledgment](#acknowledgment)
- [Reporting](#reporting)
- [Appendix](#appendix)
  - [About data](#about-data)
  - [Source Code and Dependencies](#source-code-and-dependencies)

---

# Acknowledgment  

I acknowledge the Bank Marketing Dataset, originally created by Sérgio Moro (ISCTE-IUL), Paulo Cortez (Univ. Minho), and Paulo Rita (ISCTE-IUL) in 2014. Their research on predicting the success of bank telemarketing campaigns has significantly contributed to data-driven marketing strategies in the financial sector.  

I reference their publication:  
Moro et al., 2014 – *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems. Available at:  
- [Paper](http://dx.doi.org/10.1016/j.dss.2014.03.001)  
- [BibTex](http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt)  

I also acknowledge Banco de Portugal for providing macroeconomic indicators that enriched the dataset. The dataset was accessed from the UCI Machine Learning Repository, enabling extensive research in predictive analytics and financial marketing.  
Finally, I extend my appreciation to my mentors, colleagues, and peers for their support, constructive feedback, and encouragement throughout this project.

---

# Report

**Final Data Analysis Report**

 1. Introduction
This report analyzes data from a Portuguese bank's marketing campaign focused on term deposit subscriptions. The dataset contains 41,188 client records with 21 features, including demographic, economic, and campaign-related variables. The goal is to understand client behavior, identify key factors influencing subscription decisions, and prepare data for predictive modeling.



 2. Data Understanding & Cleaning

 2.1. Dataset Overview
- Size: 41,188 rows × 21 columns (30.26 MB after optimization)
- Key Variables:
  - Numerical: age, duration, campaign, economic indicators (euribor3m, nr.employed)
  - Categorical: job, marital, education, contact, month
  - Target: y (11.27% subscribed, 88.73% did not).

 2.2. Data Quality Issues
- Missing Values: 
  - default (20.87%), education (4.2%), housing/loan (2.4% each).
  - Total missing: 1.47% of dataset.
- Duplicates: 12 duplicates removed.
- Outliers: Detected in age, duration, campaign, pdays, and cons.conf.idx.

 2.3. Cleaning Steps

- Replaced 'unknown' with NaN and imputed missing values using mode.
- Removed duplicates and irrelevant columns (duration, pdays, previous, poutcome).
- Capped outliers:
   - age > 69.5 set to 69.5.
   - campaign > 6 capped at 6.
   - Winsorized cons.conf.idx (1st–98th percentiles).



 3. Exploratory Data Analysis (EDA)

 3.1. Univariate Analysis
 Numerical Variables
- Age: Mean = 40, right-skewed (skewness = 0.78). Most clients aged 30–50.
- Campaign: Mean = 2.57 contacts/client, highly skewed (skewness = 4.76).
- Economic Indicators:
  - euribor3m (3-month interest rate): Mean = 3.62, range = 0.634–5.045.
  - nr.employed (employment count): Mean = 5,167, negatively skewed.

 Categorical Variables
- Job: Admin (26.1%), blue-collar (22.5%), technician (16.4%).
- Education: 33.7% university degrees, 23.1% high school.
- Subscription Rate: 11.27% (yes).

 3.2. Bivariate/Multivariate Analysis
 Key Correlations
- Strong Multicollinearity:
  - emp.var.rate ↔ euribor3m (ρ = 0.94)
  - emp.var.rate ↔ nr.employed (ρ = 0.94)
- Target Associations:
  - Negative: euribor3m (ρ = -0.27), nr.employed (ρ = -0.28).
  - Positive: cons.conf.idx (ρ = 0.04).

 Insights
- Economic Impact: Higher interest rates (euribor3m) and employment levels correlate with lower subscriptions.
- Demographics: Married clients and university graduates were more likely to subscribe.
- Campaign Strategy: Cellular contact outperformed landline (63.5% vs. 36.5%).



 4. Feature Engineering

 4.1. Encoding
- One-Hot Encoding: job, marital, month, day_of_week.
- Label Encoding: housing, contact.
- Ordinal Encoding: education (ordered by literacy level).

 4.2. Feature Selection
- Removed: default, loan (no statistical association with target).
- Retained: 16 features after addressing multicollinearity.



 5. Key Findings
    1. Economic Indicators Dominate: euribor3m, nr.employed, and emp.var.rate strongly influence subscriptions.
    2. Demographic Levers: Married, educated clients in stable jobs (admin/management) are prime targets.
    3. Campaign Optimization:
       - Avoid excessive contacts (>6/campaign).
       - Focus on cellular outreach and May/July/August campaigns.



 6. Recommendations
    1. Target High-Potential Groups: Prioritize married clients with university degrees.
    2. Adjust Timing: Intensify campaigns during May–August.
    3. Economic Monitoring: Align campaigns with favorable interest rate periods.
    4. Mitigate Data Leakage: Exclude post-call metrics like duration in future models.
    

---

**Final Model Comparison and Report**

 Introduction  
This report evaluates the performance of machine learning models for predicting term deposit subscriptions in a banking context. The dataset is imbalanced (8.7% positive class), emphasizing the need for models that prioritize recall (capturing subscribers) and precision (minimizing false positives). Five base models and three tuned models were compared, with LightGBM emerging as the optimal choice.  


 Model Comparison  

 Base Models vs. Tuned Models  
Key metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC) and training efficiency were analyzed:  

 Base Models Summary  
| Model             | Accuracy | Precision | Recall | F1-Score | ROC AUC | Training Time (s) |  
|-------------------|----------|-----------|--------|----------|---------|-------------------|  
| LightGBM          | 84.2%    | 38.0%     | 63.7%  | 47.6%    | 80.1%   | 0.46              |  
| RandomForest      | 83.4%    | 36.4%     | 63.4%  | 46.2%    | 80.1%   | 2.79              |  
| XGBoost           | 83.6%    | 36.3%     | 60.6%  | 45.4%    | 78.4%   | 0.50              |  

 Tuned Models Summary  
| Model             | Accuracy | Precision | Recall | F1-Score | ROC AUC | Training Time (s) |  
|-------------------|----------|-----------|--------|----------|---------|-------------------|  
| LightGBM      | 86.8%| 43.7% | 60.9%| 50.9%| 81.1%| 0.85          |  
| XGBoost           | 87.7%    | 46.4%     | 58.7%  | 51.9%    | 80.9%   | 0.93              |  
| RandomForest      | 86.8%    | 43.8%     | 60.2%  | 50.7%    | 79.7%   | 9.46              |  

Key Improvements Post-Tuning:  
- LightGBM: F1-Score improved by 6.8%, ROC AUC by 1%, with minimal overfitting.  
- XGBoost: Precision increased by 27.9%, balancing recall trade-offs.  
- RandomForest: Accuracy improved by 3.4%, but training time increased significantly.  


 Feature Importance Interpretation  
The top influential features across models include:  
1. euribor3m (Euro Interbank Offered Rate): Strongest predictor, reflecting macroeconomic conditions impacting deposit rates.  
2. age: Younger clients may prefer short-term deposits, while older clients prioritize stability.  
3. campaign: Fewer contacts during the campaign correlate with higher subscription likelihood.  
4. nr.employed (Employment Rate): Lower employment rates may drive conservative financial decisions like term deposits.  
5. cons.conf.idx (Consumer Confidence): Higher confidence aligns with investment willingness.  

Strategic Insight: Marketing should target demographics sensitive to economic trends (e.g., older clients during high euribor3m periods).  


 Best Model Justification: LightGBM  
Why LightGBM?  
1. Performance: Highest F1-Score (50.9%) and Recall (60.9%) post-tuning, ensuring balanced identification of subscribers.  
2. Efficiency: Fastest training time (0.85s) among tuned models, ideal for real-time deployment.  
3. Robustness: Minimal overfitting (Train-Test Accuracy Gap: -0.44%) and strong ROC AUC (81.1%).  
4. Interpretability: Clear feature importance aligns with domain knowledge (e.g., economic indicators).  

Threshold Optimization:  
- 0.60 Threshold: Maximizes F1-Score (51.5%), balancing precision (47.1%) and recall (57.3%).  
- At this threshold, the model captures 57% of subscribers while maintaining 92% specificity (low false positives).  


 Final Model Performance  

 Evaluation Metrics  
| Metric            | Value  |  
|--------------------|--------|  
| Accuracy           | 88.8%  |  
| Precision          | 47.1%  |  
| Recall             | 57.3%  |  
| F1-Score           | 51.5%  |  
| ROC AUC            | 81.2%  |  

 Confusion Matrix (Threshold = 0.60)  
|                     | Predicted Negative | Predicted Positive |  
|---------------------|--------------------|--------------------|  
| Actual Negative | 6,700 (TN)         | 607 (FP)           |  
| Actual Positive | 396 (FN)           | 532 (TP)           |  

 Cross-Validation Consistency  
- Mean F1-Score: 48.8% (±2.0%), indicating stable performance across folds.  

 Visual Insights  
1. ROC Curve: AUC = 81.2% confirms strong class separation.  
2. Precision-Recall Curve: Highlights trade-offs for imbalanced data.  
3. Calibration Curve: Slight overconfidence in probabilities, adjustable via post-processing.  


 Conclusion  
The LightGBM model, tuned to a 0.60 threshold, is recommended for predicting term deposit subscriptions. It balances recall (57.3%) and precision (47.1%), ensuring efficient resource allocation while capturing high-value customers. Feature insights align with economic intuition, enabling actionable marketing strategies (e.g., targeting during high euribor3m periods). The model’s speed, accuracy, and interpretability make it ideal for deployment in banking CRM systems.   
 
Model Saved As: final_lgbm_model.pkl  

---

**Strategic Suggestions for Enhancing Term Deposit Subscription Rates**

Optimize Customer Engagement:
1. Targeted Communication Strategies:
   - Personalization: Utilize insights from the predictive model to personalize communication based on customer profiles (e.g., job type, age). Tailored messages resonate better and can significantly increase engagement rates.
   - Optimal Timing: Leverage findings from the model about the best months and days for contacting customers. Prioritize outreach during times with historically higher engagement and success rates, such as March and October.

2. Enhanced Customer Segmentation:
   - Behavioral Segmentation: Segment customers not only based on demographic data but also their past interactions and responses to campaigns. This allows for more nuanced approaches and can target individuals more likely to respond positively.
   - Predictive Analytics: Continue refining the use of predictive analytics to anticipate customer needs and behavioral trends. This proactive approach can help in crafting offers that are more likely to be accepted.

Improve Campaign Efficiency:
3. Resource Allocation:
   - Focus on High-Probability Leads: Use the model's predictions to prioritize customers with a higher likelihood of subscription. Concentrating resources on these prospects can increase conversion rates and reduce marketing costs.
   - Dynamic Adjustment of Strategies: Implement a feedback loop where campaign results continually inform and adjust the predictive model. This dynamic approach ensures strategies remain relevant and effective over time.

4. Enhanced Training and Development:
   - Data Literacy: Invest in training for marketing teams to enhance their understanding of data-driven insights and model outputs. A higher level of data literacy can improve decision-making and strategic planning.
   - Continuous Learning: Encourage ongoing learning and adaptation of new analytical methods and technologies that can further enhance predictive accuracy and campaign effectiveness.

Strengthen Customer Relationships:
5. Trust and Transparency:
   - Clear Communication: Be transparent about how customer data is used for marketing purposes. Customers who trust how their data is handled are more likely to respond positively to campaigns.
   - Engagement Programs: Develop loyalty programs or workshops that educate customers about financial planning and the benefits of term deposits. These initiatives can strengthen customer relationships and improve long-term engagement.

6. Feedback Mechanisms:
   - Customer Feedback: Regularly collect feedback through surveys or interactive platforms to gauge customer satisfaction with the bank's services and communication approaches. Use this feedback to refine future campaigns.
   - Analytical Reviews: Periodically review the performance of different campaign strategies to identify what works best. This should include analyzing customer segments that showed improvements or declines in engagement to continuously refine targeting strategies.

By implementing these strategic suggestions, the bank can not only enhance the effectiveness of its term deposit marketing campaigns but also foster stronger, more trusting relationships with its customers. These strategies ensure that the bank remains adaptable, customer-centric, and competitively positioned to respond to changing market conditions and customer needs.

---

**Challenges Faced Report**

Introduction:
This report outlines the key challenges encountered during the development and deployment of predictive models aimed at enhancing the marketing strategies for term deposit subscriptions at a Portuguese bank. The analysis involved models such as LightGBM, RandomForest, XGBoost, and GradientBoosting, chosen for their advanced capabilities in handling complex datasets.

Overview of Challenges:

1. Data Imbalance:
In the development of predictive models for term deposit subscription at a Portuguese bank, several significant challenges were encountered, which required integrated strategies for effective resolution. A predominant issue was the data imbalance, where the number of non-subscribers substantially outnumbered subscribers, leading to models biased towards predicting the majority class. To counteract this, techniques such as adjusting class weights in RandomForest and using the is_unbalance option in LightGBM were employed, which helped in giving more importance to the minority class during training.

2. Feature Selection and Engineering:
Feature selection and engineering also presented a hurdle as the dataset contained a diverse range of variables, affecting the models’ performance due to overfitting or underfitting if not handled properly. Through extensive feature engineering, including encoding categorical variables and utilizing feature importance analysis specific to each model, the most impactful predictors were retained, enhancing model accuracy and generalizability.

3. Multicollinearity:
Multicollinearity among economic indicators such as emp.var.rate, euribor3m, and nr.employed was another challenge that could potentially distort model estimates and sensitivity. This was mitigated by employing tree-based models known for their robustness against multicollinearity and conducting variance inflation factor (VIF) analysis to guide the inclusion of features.

4. Model Complexity and Training, tuning Time:
Model complexity and prolonged training times, especially evident in models like GradientBoosting, posed practical limitations in rapidly changing market conditions. Optimizing model parameters to balance complexity with performance, particularly focusing on more efficient models like LightGBM, facilitated quicker training and deployment.

5. Overfitting:
Finally, initial iterations of the models showed tendencies of overfitting, where they performed exceptionally well on training data but poorly on unseen test data. Implementing cross-validation, regularization of model parameters, and subsampling techniques helped in reducing overfitting, ensuring the models maintained their predictive power on new, unseen datasets.

Conclusion:
Addressing these challenges required a blend of advanced analytical techniques and strategic decision-making to ensure that the predictive models not only perform well statistically but also align with the practical marketing needs of the bank. The resolutions implemented have significantly enhanced the models' robustness, making them valuable tools for identifying potential subscribers and optimizing marketing resources effectively. This report serves as a foundation for future projects, providing insights into overcoming similar challenges in predictive modeling within the financial sector.

---

# Author Information

- Dhanesh B. B.  

- Contact Information:  
    - [Email](dhaneshbb5@gmail.com) 
    - [LinkedIn](https://www.linkedin.com/in/dhanesh-b-b-2a8971225/) 
    - [GitHub](https://github.com/dhaneshbb)

---


# References

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, In press. DOI: [10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001).

- **Available at**:
  - [PDF](http://dx.doi.org/10.1016/j.dss.2014.03.001)
  - [BIB](http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt)

----

# Appendix

## About data
 
 Portuguese Bank Marketing PRD - Key Information

 1. Title  
Bank Marketing (with social/economic context)  

 2. Sources  
Created by: Sérgio Moro (ISCTE-IUL), Paulo Cortez (Univ. Minho), and Paulo Rita (ISCTE-IUL) @ 2014  

 3. Citation Request  
Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.*  
Decision Support Systems, In press. [DOI: 10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001).  

- Available at:  
  - [PDF](http://dx.doi.org/10.1016/j.dss.2014.03.001)  
  - [BIB](http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt)  

 4. Dataset Information  
- Objective: Predict if a customer will subscribe to a term deposit (`y`: binary "yes"/"no").  
- Enhanced Dataset: Includes five additional social/economic indicators from *Banco de Portugal*.  
- Dataset Variants:  
  1. bank-additional-full.csv (Full dataset, ordered by date).  
  2. bank-additional.csv (10% subset for computationally expensive ML algorithms).  

 5. Data Statistics  
- Number of Instances: 41,188  
- Number of Attributes: 20 input + 1 target (`y`)  
- Missing Values: Categorical attributes have missing values coded as `"unknown"`  

 6. Attribute Information  
 Bank Client Data
1. age (numeric)  
2. job (categorical: "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown")  
3. marital (categorical: "divorced", "married", "single", "unknown")  
4. education (categorical: "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown")  
5. default (categorical: "no", "yes", "unknown") - Has credit in default?  
6. housing (categorical: "no", "yes", "unknown") - Has a housing loan?  
7. loan (categorical: "no", "yes", "unknown") - Has a personal loan?  

 Last Contact Information
8. contact (categorical: "cellular", "telephone") - Contact communication type  
9. month (categorical: "jan", "feb", ..., "dec") - Last contact month  
10. day_of_week (categorical: "mon", "tue", "wed", "thu", "fri") - Last contact day of the week  
11. duration (numeric) - Last contact duration (in seconds) *(Should be excluded for real-world predictive models)*  

 Campaign & Previous Contact Information
12. campaign (numeric) - Number of contacts in the current campaign  
13. pdays (numeric) - Days since the client was last contacted (999 means never contacted)  
14. previous (numeric) - Number of previous contacts  
15. poutcome (categorical: "failure", "nonexistent", "success") - Outcome of the previous campaign  

 Social & Economic Context Attributes
16. emp.var.rate (numeric) - Employment variation rate (quarterly)  
17. cons.price.idx (numeric) - Consumer price index (monthly)  
18. cons.conf.idx (numeric) - Consumer confidence index (monthly)  
19. euribor3m (numeric) - Euribor 3-month rate (daily)  
20. nr.employed (numeric) - Number of employees (quarterly)  

 Target Variable
21. y (binary: "yes", "no") - Client subscribed to a term deposit?

## Source Code and Dependencies

In the development of the predictive models for the term deposit subscription project at the Portuguese bank, I extensively utilized several functions from my custom library "insightfulpy." This library, available on both GitHub and PyPI, provided crucial functionalities that enhanced the data analysis and modeling process. For those interested in exploring the library or using it in their own projects, you can inspect the source code and documentation available. The functions from "insightfulpy" helped streamline data preprocessing, feature engineering, and model evaluation, making the analytic processes more efficient and reproducible.

You can find the source and additional resources on GitHub here: [insightfulpy on GitHub](https://github.com/dhaneshbb/insightfulpy), and for installation or further documentation, visit [insightfulpy on PyPI](https://pypi.org/project/insightfulpy/). These resources provide a comprehensive overview of the functions available and instructions on how to integrate them into your data science workflows.

---


Below is an overview of each major tool (packages, user-defined functions, and imported functions) that appears in this project.

<pre>
Imported packages:
1: builtins
2: builtins
3: pandas
4: warnings
5: researchpy
6: matplotlib.pyplot
7: missingno
8: seaborn
9: numpy
10: scipy.stats
11: textwrap
12: logging
13: statsmodels.api
14: time
15: xgboost
16: lightgbm
17: catboost
18: scikitplot
19: psutil
20: os
21: gc
22: joblib
23: types
24: inspect

User-defined functions:
1: memory_usage
2: dataframe_memory_usage
3: garbage_collection
4: chi_square_test
5: fisher_exact_test
6: spearman_correlation_with_target
7: hypothesis_testing_mann_whitney
8: cap_outliers_percentile
9: normality_test_with_skew_kurt
10: spearman_correlation
11: calculate_vif
12: evaluate_model
13: tune_hyperparameters
14: threshold_analysis
15: cross_validation_analysis_table
16: plot_all_evaluation_metrics
17: show_default_feature_importance

Imported functions:
1: open
2: tabulate
3: display
4: is_datetime64_any_dtype
5: skew
6: kurtosis
7: shapiro
8: kstest
9: compare_df_columns
10: linked_key
11: display_key_columns
12: interconnected_outliers
13: grouped_summary
14: calc_stats
15: iqr_trimmed_mean
16: mad
17: comp_cat_analysis
18: comp_num_analysis
19: detect_mixed_data_types
20: missing_inf_values
21: columns_info
22: cat_high_cardinality
23: analyze_data
24: num_summary
25: cat_summary
26: calculate_skewness_kurtosis
27: detect_outliers
28: show_missing
29: plot_boxplots
30: kde_batches
31: box_plot_batches
32: qq_plot_batches
33: num_vs_num_scatterplot_pair_batch
34: cat_vs_cat_pair_batch
35: num_vs_cat_box_violin_pair_batch
36: cat_bar_batches
37: cat_pie_chart_batches
38: num_analysis_and_plot
39: cat_analyze_and_plot
40: chi2_contingency
41: fisher_exact
42: pearsonr
43: spearmanr
44: ttest_ind
45: mannwhitneyu
46: linkage
47: dendrogram
48: leaves_list
49: variance_inflation_factor
50: train_test_split
51: cross_val_score
52: learning_curve
53: resample
54: compute_class_weight
55: accuracy_score
56: precision_score
57: recall_score
58: f1_score
59: roc_auc_score
60: confusion_matrix
61: precision_recall_curve
62: roc_curve
63: auc
64: calibration_curve
65: classification_report
</pre>