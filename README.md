# Portuguese Bank Marketing Campaign Analysis

## Project Overview
This project analyzes direct phone marketing campaigns by a Portuguese bank (2008-2010) to predict term deposit subscriptions. It combines **exploratory data analysis**, **predictive modeling**, and **strategic recommendations** to optimize customer targeting. The workflow includes:

1. **Data Analysis**: Exploratory insights into customer demographics, economic indicators, and campaign performance.
2. **Predictive Modeling**: Development of machine learning models (LightGBM, XGBoost, RandomForest) to identify high-potential customers.
3. **Actionable Recommendations**: Data-driven strategies to enhance campaign effectiveness.

## Dataset
**Domain**: Banking/Finance  
**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)  
**Enriched Data**: Includes macroeconomic indicators from Banco de Portugal.  
**Size**: 41,188 records × 21 features.  

### Key Features
- **Demographics**: Age, job, marital status, education.  
- **Financial**: Credit default, housing/personal loans.  
- **Campaign Metrics**: Contact type, duration, month/day of contact.  
- **Economic Indicators**: Euribor rate, employment rate, consumer confidence index.  
- **Target**: `y` (binary: "yes"/"no"). 

**Dataset Link**: [Download Here](https://d3libtxjj3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1000-ProtugeseBank.zip)

---

## Repository Structure
```
├── data/                 # Raw and processed datasets
├── docs/                # Project briefs and dataset documentation
├── notebook/            # Jupyter notebook for analysis and modeling
├── report/              # Final report and strategic suggestions
├── results/             # Visualizations, model outputs, and analysis reports
├── scripts/             # Utility scripts and helper functions
└── requirements.txt     # Python dependencies
```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone [REPO_URL]
   cd PROTUGESEBANK
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Custom Library**: Install `insightfulpy` for streamlined analytics:
   ```bash
   pip install insightfulpy
   ```

---

## Usage
### Running the Analysis
1. Open the Jupyter notebook:  
   `notebook/PRCP-1000-PortugeseBank.ipynb`
2. Execute cells sequentially to:
   - Perform data cleaning and EDA.
   - Train and evaluate predictive models.
   - Generate visualizations and reports.

### Key Outputs
- **Final Model**: `results/model/final_lgbm_model.pkl` (LightGBM with 88.8% accuracy).
- **Reports**:  
  - `report/Final Report.md`: Comprehensive analysis and recommendations.
  - `results/365csv pre-anlysis/`: Preprocessing reports, visualizations, and statistical summaries.

---

## Model Performance
| Model      | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|------------|----------|-----------|--------|----------|---------|
| LightGBM   | 88.8%    | 47.1%     | 57.3%  | 51.5%    | 81.2%   |
| XGBoost    | 87.7%    | 46.4%     | 58.7%  | 51.9%    | 80.9%   |
| RandomForest | 86.8%  | 43.8%     | 60.2%  | 50.7%    | 79.7%   |

## Challenges Faced
1. **Data Imbalance**: Addressed using class weighting and LightGBM’s `is_unbalance` parameter.
2. **Multicollinearity**: reduced complexity Mitigated via VIF analysis and tree-based models.
3. **Overfitting**: Controlled through cross-validation and regularization.

---

## References

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, In press. DOI: [10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001).

- **Available at**:
  - [PDF](http://dx.doi.org/10.1016/j.dss.2014.03.001)
  - [BIB](http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt)

---

**License**: This project is for educational/research purposes. Cite the dataset authors when referencing this work.
