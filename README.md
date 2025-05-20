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
  ![image](https://github.com/user-attachments/assets/95606ca6-f961-48d9-8c0a-98c69171653b) <br />

- First retrieve the list of columns available in the CSV file. Use the following to import the required libraries and print the list of columns
  ```
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
  import matplotlib.pyplot as plt
  import seaborn as sns
  df = pd.read_csv(".\Telco-Customer-Churn-Dataset.csv")
  print(df.columns.tolist())
  ```
  ![image](https://github.com/user-attachments/assets/2d841f83-7989-4222-b0bd-ea8c2ae82c10) <br />

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
  ![image](https://github.com/user-attachments/assets/371603fd-c5b0-48fb-b4c4-24954b6717c5) <br />

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
  ![image](https://github.com/user-attachments/assets/dc1faa3d-2d91-4b0b-909f-debcc61f8198) <br />


## Model Selection, Training and Evaluation
- Import the required libraries
  ```

  ```
  ![image](https://github.com/user-attachments/assets/730bd478-c2da-4d3e-b9b2-8aeffb2e5d29) <br />


- Define features and labels
  ```
  df = pd.get_dummies(df, columns=['Churn'], drop_first=True)

  # Assume df is your processed DataFrame with encoded columns
  X = df.drop('Churn_Yes', axis=1)  # 'Churn_Yes' is the result of one-hot encoding
  y = df['Churn_Yes']
  ```

- Train-test split
  ```
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
  )
  ```

- Scale the data for models that require it
  ```
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```

- Train the models
  ```
  models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
  }
  
  trained_models = {}
  
  for name, model in models.items():
      if name in ["Logistic Regression", "SVM"]:
          model.fit(X_train_scaled, y_train)
      else:
          model.fit(X_train, y_train)
      trained_models[name] = model
  ```

- Evaluate the function
  ```
  def evaluate_model(name, model, X_test, y_test):
    if name in ["Logistic Regression", "SVM"]:
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
  ```

- Evaluate all the models
  ```
  results = {}
  
  for name, model in trained_models.items():
      results[name] = evaluate_model(name, model, X_test, y_test)
  
  results_df = pd.DataFrame(results).T.sort_values(by="F1 Score", ascending=False)
  print(results_df)
  ```

- Confusion Matrices
  ```
  for name, model in trained_models.items():
    if name in ["Logistic Regression", "SVM"]:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix: {name}')
    plt.show()
  ```


- Complete Python code
  ```
  # Customer Churn Prediction Using Supervised Learning
  
  # 1. Import Libraries
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  
  from sklearn.metrics import (
      accuracy_score,
      precision_score,
      recall_score,
      f1_score,
      roc_auc_score,
      confusion_matrix,
      ConfusionMatrixDisplay
  )
  
  # 2. Load Dataset
  df = pd.read_csv('./Telco-Customer-Churn-Dataset.csv')
  
  # 3. Drop irrelevant columns
  df.drop('customerID', axis=1, inplace=True)
  
  # 4. Convert target variable to binary
  df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
  
  # 5. Handle missing or bad data in TotalCharges
  df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
  df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
  
  # 6. One-hot encode categorical variables (drop_first to avoid dummy trap)
  df = pd.get_dummies(df, drop_first=True)
  
  # 7. Define features and target
  X = df.drop('Churn', axis=1)
  y = df['Churn']
  
  # 8. Train-test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, stratify=y, random_state=42
  )
  
  # 9. Standardize features for models that require it
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # 10. Initialize models
  models = {
      "Logistic Regression": LogisticRegression(max_iter=1000),
      "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
      "SVM": SVC(probability=True)
  }
  
  # 11. Train models
  trained_models = {}
  
  for name, model in models.items():
      if name in ["Logistic Regression", "SVM"]:
          model.fit(X_train_scaled, y_train)
      else:
          model.fit(X_train, y_train)
      trained_models[name] = model
  
  # 12. Evaluation function
  def evaluate_model(name, model, X_test, y_test):
      if name in ["Logistic Regression", "SVM"]:
          y_pred = model.predict(X_test_scaled)
          y_proba = model.predict_proba(X_test_scaled)[:, 1]
      else:
          y_pred = model.predict(X_test)
          y_proba = model.predict_proba(X_test)[:, 1]
  
      return {
          "Accuracy": accuracy_score(y_test, y_pred),
          "Precision": precision_score(y_test, y_pred),
          "Recall": recall_score(y_test, y_pred),
          "F1 Score": f1_score(y_test, y_pred),
          "ROC-AUC": roc_auc_score(y_test, y_proba)
      }
  
  # 13. Evaluate all models
  results = {}
  
  for name, model in trained_models.items():
      results[name] = evaluate_model(name, model, X_test, y_test)
  
  results_df = pd.DataFrame(results).T.sort_values(by="F1 Score", ascending=False)
  print("\nModel Evaluation Results:")
  print(results_df)
  
  # 14. Confusion Matrices
  for name, model in trained_models.items():
      if name in ["Logistic Regression", "SVM"]:
          y_pred = model.predict(X_test_scaled)
      else:
          y_pred = model.predict(X_test)
  
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot()
      plt.title(f'Confusion Matrix: {name}')
      plt.show()
  ```


