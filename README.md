# Predicting Employee Attrition Using Machine Learning

![Pipeline Flowchart](pipeline_flowchart.png)

## Overview
This project presents an end-to-end Machine Learning pipeline designed to predict employee attrition using the **IBM HR Analytics dataset**. Built as part of the **7005SCN Individual Research Project (MSc Data Science, Coventry University)**, the project addresses class imbalance, model interpretability, and algorithmic fairness to provide actionable insights for HR decision-making.

---

## Key Features & Workflow

1. **Exploratory Data Analysis (EDA)**
   - Visualized attrition distribution across demographic, operational, and financial indicators.
   - Identified critical factors influencing turnover (e.g., OverTime, MonthlyIncome, TotalWorkingYears).

2. **Data Preprocessing & Class Imbalance Handling**
   - Encoded categorical features and scaled numerical variables.
   - Implemented **SMOTE** and hybrid resampling strategies to address severe class imbalance.

3. **Machine Learning Modeling & Evaluation**
   - Trained and benchmarked multiple algorithms:
     - **Logistic Regression**
     - **Random Forest**
     - **XGBoost**
   - Evaluated models using **Precision, Recall, F1-Score, ROC-AUC**, and confusion matrices to minimize false negatives (unidentified at-risk employees).

4. **Model Explainability (XAI)**
   - Utilized **SHAP (SHapley Additive exPlanations)** values (bar plots & beeswarm summary) to uncover global and feature-level drivers of attrition.

5. **Algorithmic Fairness & Ethics Assessment**
   - Evaluated model predictions across demographic subgroups to detect potential bias and ensure ethical HR analytics.

---

## Project Structure

```text
.
├── 7005SCN_CW2_Attrition_Project.ipynb  # Primary Jupyter Notebook (Code & Analysis)
├── 7005SCN_CW2_Project_Report.docx      # Comprehensive Research Report
├── pipeline_flowchart.png               # High-level ML pipeline architectural diagram
├── fig1_attrition_distribution.png       # EDA: Target class distribution
├── fig2_attrition_by_category.png        # EDA: Categorical feature breakdown
├── fig3_numerical_distributions.png     # EDA: Numerical features
├── fig4_correlation_heatmap.png         # Feature correlation analysis
├── fig5_smote_effect.png                # Imbalance handling analysis
├── fig6_model_comparison.png            # Model benchmark results
├── fig7_confusion_matrices.png          # Model evaluation matrices
├── fig8_roc_curves.png                  # ROC-AUC curves
├── fig9_shap_beeswarm.png               # SHAP feature importance beeswarm
├── fig10_shap_bar.png                   # SHAP global feature impact
├── fig11_fairness.png                   # Fairness & ethical impact evaluation
└── README.md                            # Project documentation
