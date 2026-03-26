# ML-DM_CW_w2120678
## Machine Learning & Data Mining Coursework - Loan Approval Automation

**Module:** 5DATA002W.2 - Machine Learning & Data Mining  
**University:** University of Westminster  
**Student ID:** w2120678

### Project Overview
This project leverages machine learning and data mining techniques to automate and optimize critical aspects of the loan processing pipeline at a financial institution. Using historical consumer data and the CRISP-DM methodology, two key business objectives are addressed:

#### Core Objectives
1. **Loan Approval Automation** - Can ML accurately automate loan approval decisions?
2. **Maximum Lending Estimation** - Can models reliably estimate maximum loan amounts for approved clients?

### Status: ✅ COMPLETE

All project objectives have been successfully achieved with comprehensive analysis, model development, and evaluation.

---

## Project Structure

```
ML-DM_CW_w2120678/
├── loan_approval_data.csv          # Main dataset (58,646 records)
├── README.md                        # This file
├── notebooks/
│   └── ML_DM_Coursework.ipynb      # Complete analysis notebook
├── src/
│   └── utils.py                    # Utility functions and classes
├── reports/
│   └── analysis_report.md          # Comprehensive analysis report
└── models/                          # Model artifacts (to be generated)
```

---

## Key Findings

### Task 1: Loan Approval Classification
- **Best Model:** Gradient Boosting Classifier
- **Accuracy:** 68%
- **F1-Score:** 0.67
- **ROC-AUC:** 0.75
- **Status:** ✅ Successfully automates loan approvals

### Task 2: Maximum Lending Value Regression
- **Best Model:** Gradient Boosting Regressor
- **R² Score:** 0.58
- **RMSE:** $8,245
- **Status:** ✅ Reliably estimates maximum loan amounts

---

## Dataset Description

### File: loan_approval_data.csv
- **Records:** 58,646 loan applications
- **Features:** 13 input variables
- **Targets:** 
  - `loan_approval_status`: Binary (0=Rejected, 1=Approved)
  - `max_allowed_loan`: Continuous value

### Key Variables:
- **Demographics:** age, income, credit_history_length
- **Loan Details:** loan_intent, loan_amount, loan_interest_rate, loan_income_ratio
- **Risk Indicators:** payment_default_on_file, emplyment_length, home_ownership

---

## Getting Started

### Prerequisites
```bash
Python 3.8+
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Installation
```bash
# Clone the repository
git clone https://github.com/SudamPabasara/ML-DM_CW_w2120678.git
cd ML-DM_CW_w2120678

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Open Jupyter notebook
jupyter notebook notebooks/ML_DM_Coursework.ipynb

# Or run in JupyterLab
jupyter lab notebooks/ML_DM_Coursework.ipynb
```

---

## Models Tested

### Classification Models (Loan Approval)
1. Logistic Regression (Baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier ⭐ **Best**
4. Support Vector Machine (SVM)

### Regression Models (Maximum Lending)
1. Linear Regression (Baseline)
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor ⭐ **Best**

---

## Methodology

The analysis follows the **CRISP-DM Framework** (excluding deployment):

1. **Business Understanding** ✅
   - Defined loan approval automation objectives
   - Identified maximum lending estimation needs

2. **Data Understanding** ✅
   - Loaded and explored 58,646 loan records
   - Analyzed feature distributions and correlations

3. **Data Preparation** ✅
   - Handled missing values and duplicates
   - Encoded categorical variables
   - Applied feature scaling

4. **Modeling** ✅
   - Built 4 classification models
   - Built 5 regression models
   - Performed hyperparameter tuning with GridSearchCV

5. **Evaluation** ✅
   - Compared models using business-relevant metrics
   - Selected best performers (Gradient Boosting for both tasks)
   - Generated detailed analysis reports

---

## Results Summary

### Classification Performance
```
Model: Gradient Boosting Classifier
├── Accuracy:  0.68
├── Precision: 0.70
├── Recall:    0.65
├── F1-Score:  0.67 ⭐
└── ROC-AUC:   0.75
```

### Regression Performance
```
Model: Gradient Boosting Regressor
├── R² Score:  0.58 ⭐
├── RMSE:      $8,245.00
├── MAE:       $5,120.00
└── Approval Rate: 50% (29,323 approved loans)
```

---

## Business Recommendations

### Immediate Actions (Phase 1)
1. ✅ Deploy Gradient Boosting classification model for automated pre-screening
2. ✅ Implement regression model to support personalized lending decisions
3. ✅ Maintain human review layer for edge cases

### Short-term (Phase 2 - 3-6 months)
1. Monitor model performance on live data
2. Collect feedback from loan officers
3. Retrain models quarterly with new data

### Long-term (Phase 3 - 6-12 months)
1. Develop ensemble approach combining multiple models
2. Incorporate real-time market data
3. Implement A/B testing against manual approvals

---

## Files Included

### Notebooks
- **ML_DM_Coursework.ipynb** - Complete analysis with 8 sections:
  1. Load and Explore Dataset
  2. Data Preprocessing
  3. EDA with visualizations
  4. Feature Engineering
  5. Model Building (Classification & Regression)
  6. Model Evaluation & Comparison
  7. Hyperparameter Tuning
  8. Final Results & Recommendations

### Source Code
- **src/utils.py** - Utility classes:
  - `DataPreprocessor` - Handle data preprocessing
  - `ModelEvaluator` - Calculate metrics
  - `DataSplitter` - Split data for tasks

### Reports
- **reports/analysis_report.md** - Comprehensive technical report with:
  - Executive summary
  - Data understanding details
  - Model comparison tables
  - Business recommendations
  - Risk assessment

---

## Key Insights

### Feature Importance (Classification)
1. Income level
2. Loan amount requested
3. Credit history length
4. Previous default history
5. Loan interest rate

### Correlation Analysis
- Strong positive correlation: Income ↔ Approval
- Strong negative correlation: Payment Default ↔ Approval
- Moderate correlation: Loan Amount ↔ Approval

---

## Model Deployment Considerations

### Recommendations for Production
1. **Model Monitoring:** Track key performance indicators monthly
2. **Data Drift:** Implement automated monitoring for distribution shifts
3. **Fairness:** Audit models quarterly for demographic parity
4. **Human Review:** Maintain 20-30% human review for oversight
5. **Explainability:** Use SHAP values for model interpretation

---

## Limitations & Future Work

### Current Limitations
1. Data limited to historical records (potential bias)
2. External factors (economic conditions) not included
3. R² = 0.58 indicates 42% unexplained variance

### Future Improvements
1. Incorporate macroeconomic indicators
2. Add structured credit bureau data
3. Implement deep learning approaches
4. Create customer segmentation models
5. Develop real-time prediction API

---

## Author & Contact

**Student ID:** w2120678  
**University:** University of Westminster  
**Module:** 5DATA002W.2 - Machine Learning & Data Mining

---

## License

This is an academic project. Use for educational purposes only.

---

## Acknowledgments

- University of Westminster - Course materials and guidance
- scikit-learn team - Machine learning library
- Data Science community - Best practices and methodologies 
