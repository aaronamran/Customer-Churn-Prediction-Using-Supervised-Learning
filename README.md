# Customer Churn Prediction Using Supervised Learning
Jupyter Notebook is used for this project, and the final code will be compiled into a single Python file
- Objective: Predict whether a customer will churn (yes/no) next period
- Business impact: Reducing churn by identifying at-risk customers early
- Success criteria: A model with strong recall (to catch most churners) and balanced precision (to avoid too many false alarms), e.g. F1-score ≥ 0.7 or a specific lift over baseline

![customer_churn_prediction_supervised_learning](https://github.com/user-attachments/assets/7921b612-67b4-46a8-bc87-99ce90b4024f)

1. [Exploratory Data Analysis and Data Preprocessing](#exploratory-data-analysis-and-data-preprocessing)
2. [Model Selection, Training and Evaluation](#model-selection-training-and-evaluation)



## Exploratory Data Analysis and Data Preprocessing

- In this Kaggle website called [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data), download the Telco Customer Churn dataset as a ZIP file <br />
  ![image](https://github.com/user-attachments/assets/3ccb8762-89a8-41dc-bed3-2a56b534aadc) <br />
  ![image](https://github.com/user-attachments/assets/b4d1d85d-5040-4ffb-acd2-f05a9c65d53b) <br />

- Extract the ZIP file. It will only contain 1 CSV file as shown <br />
  ![image](https://github.com/user-attachments/assets/2813a7fd-9e27-4b0d-a23a-3698a8331e2a) <br />

  In Jupyter Notebook, it looks like this <br />
  ![image](https://github.com/user-attachments/assets/68a1b407-5baf-425b-a348-8e5e1048e173) <br />

- First retrieve the list of columns available in the CSV file. Use
  ```
  import pandas as pd
  df = pd.read_csv(".\Telco-Customer-Churn-Dataset.csv")
  print(df.columns.tolist())
  ```
  ![image](https://github.com/user-attachments/assets/97a427f6-68bd-4da3-86cc-68ce8db95366) <br />

- Drop the columns that do not help and convert the target variable into binary
  ```
  df.drop('customerID', axis=1, inplace=True)
  df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
  ```

- Handle the different data types and missing values
  ```
  df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # turns bad values into NaN
  df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())  # fill missing with median
  ```
  ![image](https://github.com/user-attachments/assets/6226c75d-9a82-445a-a28e-613f6970f051) <br />


- The current status of the columns are
  - Numerical: SeniorCitizen — (0/1), tenure, MonthlyCharges, TotalCharges

  - Categorical (need encoding): gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
 
- Encode the target column
  ```
  df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
  ```
 
- Encode the categorical features. For simplicity, use one-hot encoding to turn categorical columns into binary ones like Contract_Month-to-month, gender_Male, etc.
  ```
  df = pd.get_dummies(df, drop_first=True)
  ```

- Feature Scaling (for distance-based models)





## Model Selection, Training and Evaluation
- Try a range of classifiers:

Logistic Regression

Decision Tree / Random Forest

k-Nearest Neighbors

Gradient Boosting (e.g. XGBoost, LightGBM)

evaluation:
- Use metrics that capture both recall and precision:

Confusion matrix

Precision, Recall, F1-score

ROC AUC

PR AUC (especially important if classes are imbalanced)
