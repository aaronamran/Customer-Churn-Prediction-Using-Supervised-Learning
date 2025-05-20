# Customer Churn Prediction Using Supervised Learning
Jupyter Notebook is used for this project, and the final code will be compiled into a single Python file
- Objective: Predict whether a customer will churn (yes/no) next period
- Business impact: Reducing churn by identifying at-risk customers early
- Success criteria: A model with strong recall (to catch most churners) and balanced precision (to avoid too many false alarms), e.g. F1-score ≥ 0.7 or a specific lift over baseline

![customer_churn_prediction_supervised_learning](https://github.com/user-attachments/assets/7921b612-67b4-46a8-bc87-99ce90b4024f)


- In this Kaggle website called [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data), download the Telco Customer Churn dataset as a ZIP file <br />
  ![image](https://github.com/user-attachments/assets/3ccb8762-89a8-41dc-bed3-2a56b534aadc) <br />
  ![image](https://github.com/user-attachments/assets/b4d1d85d-5040-4ffb-acd2-f05a9c65d53b) <br />

- Extract the ZIP file. It contains only 1 CSV file as shown <br />
  ![image](https://github.com/user-attachments/assets/2813a7fd-9e27-4b0d-a23a-3698a8331e2a) <br />

  In Jupyter Notebook, it looks like this <br />
  ![image](https://github.com/user-attachments/assets/95606ca6-f961-48d9-8c0a-98c69171653b) <br />

- The complete Python code used is below
  ```
  # Customer Churn Prediction Using Supervised Learning
  
  # 1. Import Libraries
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  import os
  
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
  
  # 14. Confusion Matrices and Save Plots
  # Create a directory to store plots if it doesn't exist
  os.makedirs("plots", exist_ok=True)
  
  for name, model in trained_models.items():
      if name in ["Logistic Regression", "SVM"]:
          y_pred = model.predict(X_test_scaled)
      else:
          y_pred = model.predict(X_test)
  
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.title(f'Confusion Matrix: {name}')
  
      # Save the figure before showing it
      filename = f"plots/confusion_matrix_{name.replace(' ', '_')}.png"
      plt.savefig(filename, dpi=300, bbox_inches='tight')
      plt.show()
  ```

- In the '1. Import Libraries' section, all the necessary libraries are imported. `pandas` and `numpy` are used for data manipulation and numerical computations, `matplotlib` and `seaborn` are used for visualisations, and `os` is used for handling file paths and directories. The `sklearn` modules are for model training, data preprocessing and performance metrics
  ```
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  import os
  
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
  ```
- Sections 2 and 3 are for loading the dataset and dropping irrelevant columns. The `customerID` column is removed since it is not useful for modeling as it is just an identifier and does not influence the churn behaviour
  ```
  # 2. Load Dataset
  df = pd.read_csv('./Telco-Customer-Churn-Dataset.csv')
  
  # 3. Drop irrelevant columns
  df.drop('customerID', axis=1, inplace=True)
  ```

- 


  
- The visualisation output is seen below <br />
  ![image](https://github.com/user-attachments/assets/d1fe67fd-18c9-46c7-bd22-8a5c7973d954) <br />
  ![image](https://github.com/user-attachments/assets/c980ed67-7751-4661-ab0c-54c1d29df24b) <br />
  ![image](https://github.com/user-attachments/assets/ec5ef8f1-4f17-4264-8076-57b40ed11699) <br />


- The created plots will be saved as individual image files under the created folder called 'plots' <br />
  ![image](https://github.com/user-attachments/assets/e43b4639-0573-4ae5-b430-b44d72c59f4b) <br />



