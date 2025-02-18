# ATM Cash-Out Prediction ML Project
1. **Problem Understanding & Data Collection**
The goal is to predict whether an ATM will run out of cash based on past transaction patterns.

**Data includes:**
* ATM ID, Date, Withdrawal & Deposit Amounts, Total Cash, Transaction Count, and Day of the Week.

* Target variable (Cash_Out): 1 if the ATM’s remaining cash is below 20K, else 0.

2. **Exploratory Data Analysis (EDA)**
Load the dataset and check for missing values, outliers, and distributions.

* Visualize: Cash-out trends over time.
* Daily transaction patterns (e.g., weekends vs. weekdays).
* High withdrawal times affecting cash-out.

3.**Feature Engineering**

* Extract useful time-based features (e.g., day of the month, holidays).
* Create rolling averages for transaction patterns.
* Convert categorical features (e.g., Day_of_Week) into numerical format.

4. **Train-Test Split & Data Preprocessing**
* Split the dataset into training **(80%)** and testing **(20%)** sets.
* Normalize numerical features for better model performance.

5. **Model Selection & Training**
* Try different models:
    *   Logistic Regression (Baseline)
    * Random Forest (Better generalization) 
    * XGBoost (Best for imbalanced data)
* Use cross-validation to fine-tune hyperparameters.

6. **Model Evaluation**
*   Use metrics:
    * Accuracy, Precision, Recall, F1-score (for classification).
    * Confusion Matrix to check false positives/negatives.
    * ROC-AUC Curve for model performance.

