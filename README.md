# Customer Churn Prediction Using Supervised Learning

This data science project presents a supervised learning pipeline for predicting customer churn using Python and Jupyter Notebook. Based on a real-world telecom dataset, it aims to train models that accurately identify customers likely to leave. Early churn detection is vital for businesses to retain users and reduce revenue loss. The dataset is cleaned and preprocessed through feature encoding, missing value handling, and standardization. Three classifiers—Logistic Regression, Random Forest, and Support Vector Machine—are trained and evaluated. The project emphasizes high recall to catch most churners while balancing precision, using the F1-score as the main metric. Confusion matrices and ROC-AUC scores aid model comparison and interpretation. The final code is modular and compiled into a standalone Python file for easy reuse <br />
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

- The complete Python code used is below. Alternatively, the Python code file can be obtained from [here](https://github.com/aaronamran/Customer-Churn-Prediction-Using-Supervised-Learning/blob/main/customer-churn-prediction-supervised-learning.py)
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

- In the first section, all the necessary libraries are imported. `pandas` and `numpy` are used for data manipulation and numerical computations, `matplotlib` and `seaborn` are used for visualisations, and `os` is used for handling file paths and directories. The `sklearn` modules are for model training, data preprocessing and performance metrics
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

- The code in section 4 converts target variable to binary, specifically converting the target column `Churn` from categorical values (Yes and No) into corresponding binary numeric values (1 and 0), which is required for machine learning models
  ```
  # 4. Convert target variable to binary
  df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
  ```

- To handle missing or bad data in `TotalCharges`, section 5 code converts `TotalCharges` to a numeric data type. It replaces any non-convertible values (turned into NaN) with the median of the column to handle missing or invalid data
  ```
  # 5. Handle missing or bad data in TotalCharges
  df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
  df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
  ```

- To avoid multicollinearity (dummy variable trap), section 6 code converts categorical variables into numeric binary variables (dummy variables), excluding one category per feature (`drop_first=True`). Multicollinearity occurs when two or more independent variables in a dataset are highly correlated, making it difficult for a model to determine the individual effect of each variable on the target
  ```
  # 6. One-hot encode categorical variables (drop_first to avoid dummy trap)
  df = pd.get_dummies(df, drop_first=True)
  ```

- Section 7 code defines the features and target by splitting the dataset into `X` for features (independent variables) and `Y` for target variable (dependent variable - `Churn`)
  ```
  # 7. Define features and target
  X = df.drop('Churn', axis=1)
  y = df['Churn']
  ```

- The code for section 8 splits the dataset into training (80%) and testing (20%) sets while preserving the class distribution of `Churn` using `stratify=y`. The 80-20 rule is a common practice to allocate 80% of data for training the model and 20% for testing to ensure there's enough data to both learn the patterns and objectively evaluate performance on unseen data to balancing bias and variance. While 80-20 is common, the split ratio isn't fixed; it can be adjusted based on dataset size (e.g., 70-30, 90-10, etc.)
  ```
  # 8. Train-test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, stratify=y, random_state=42
  )
  ```

- Applies standardization (mean=0, std=1) to numerical features. Required for algorithms like Logistic Regression and SVM to perform optimally
  ```  
  # 9. Standardize features for models that require it
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```

- Model initialisation is done in section 10, which creates a dictionary of machine learning models. Logistic regression is a linear model for binary classification, random forest is an ensemble of decision trees from robust to noise and SVM (Support Vector Machines) is used for classification with decision boundaries
  ```
  # 10. Initialize models
  models = {
      "Logistic Regression": LogisticRegression(max_iter=1000),
      "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
      "SVM": SVC(probability=True)
  }
  ```

- Section 11 fits each model to the training data. It uses scaled features for Logistic Regression and SVM and uses unscaled features for Random Forest (tree-based models don’t need scaling)
  ```
  # 11. Train models
  trained_models = {}
  
  for name, model in models.items():
      if name in ["Logistic Regression", "SVM"]:
          model.fit(X_train_scaled, y_train)
      else:
          model.fit(X_train, y_train)
      trained_models[name] = model
  ```

- The evaluation function in section 12 is to evaluate a model using several classification metrics, which are:
  - Accuracy (Overall correctness)
  - Precision (percentage of predicted churns that were actually churns)
  - Recall (percentage of actual churns correctly predicted)
  - F1 Score (Harmonic mean of precision and recall which is useful whena balance between false positives and false negatives is required). The formula is: <br />
    F1 Score = 2 * ((Precision * Recall)/(Precision + Recall))
  - ROC-AUC (Receiver Operating Characteristic - Area Under Curve) (Ability to separate classes across all threshold values; a higher AUC means the model is better at distinguishing between churn and non-churn customers). A perfect classifier is AUC of 1.0, but AUC of 0.5 means it is no better than random guessing
  ```
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
  ```

- To evaluate all models, section 13 code loops through all the trained models, evaluates and stores the performance results and creates a DataFrame for easy comparison, which is sorted by F1 score
  ```
  # 13. Evaluate all models
  results = {}
  
  for name, model in trained_models.items():
      results[name] = evaluate_model(name, model, X_test, y_test)
  
  results_df = pd.DataFrame(results).T.sort_values(by="F1 Score", ascending=False)
  print("\nModel Evaluation Results:")
  print(results_df)
  ```

- Finally, section 14 generates and displays the confusion matrix for each model. It saves the plot images in a folder named 'plots' (this folder is created if it does not exist). The confusion matrix helps understand model errors (False Positives and Negatives)
  ```
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
  The created plots are saved as individual PNG image files under the created folder called 'plots' <br />
  ![image](https://github.com/user-attachments/assets/e43b4639-0573-4ae5-b430-b44d72c59f4b) <br />
  
  
- The visualisation output is seen below <br />
  ![image](https://github.com/user-attachments/assets/d1fe67fd-18c9-46c7-bd22-8a5c7973d954) <br />
  ![image](https://github.com/user-attachments/assets/c980ed67-7751-4661-ab0c-54c1d29df24b) <br />
  ![image](https://github.com/user-attachments/assets/ec5ef8f1-4f17-4264-8076-57b40ed11699) <br />

- Based on the F1 Score, which balances precision and recall, the models performed as follows:
  - Logistic Regression: Achieved the highest F1 score (0.609195), indicating a good balance between precision and recall. It also has the highest ROC-AUC (0.841585), suggesting strong discriminative ability
  - SVM: Has a lower F1 score (0.556231) compared to Logistic Regression, and a lower ROC-AUC (0.796047)
  - Random Forest: Also has a lower F1 score (0.550075) and ROC-AUC (0.825081) than Logistic Regression

- The interpretations of the visualised plots are as follows:
  - Logistic Regression: Shows a good balance with a relatively high number of correctly predicted churn cases (212) while maintaining a reasonable number of false positives (110)
  - Random Forest: Has a slightly lower number of correctly identified churn cases (184) and a higher number of false negatives (190) compared to Logistic Regression
  - SVM: Similar to Random Forest, SVM also has a lower number of correctly identified churn cases (183) and a higher number of false negatives (191) than Logistic Regression

- As a conclusion, Logistic Regression appears to be the best-performing model based on these metrics, offering a better balance between precision and recall, and a higher ability to discriminate between churn and no-churn cases



