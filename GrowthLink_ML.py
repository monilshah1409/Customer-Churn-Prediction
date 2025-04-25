import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
importances = []

# Load the dataset
df = pd.read_csv('/Users/Monil_149/Desktop/GrowthLink_ML/Churn_Modelling.csv')

# Exploratory Data Analysis (EDA)
def eda(df, do_plot=True):
    print("Data Overview:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    if do_plot:
        # Visualize the distribution of the target variable
        sns.countplot(x='Exited', data=df)
        plt.title('Churn Distribution')
        plt.show()
        plt.close()

        # Visualize correlations
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        plt.close()

eda(df, do_plot=False)  # Set do_plot=False to skip plotting for faster runs

# Data Preprocessing
# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))

# Convert categorical variables to numerical
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature Engineering
df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['Products_Per_Tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
df['Credit_Balance_Score'] = df['CreditScore'] * (df['Balance'] / df['EstimatedSalary'].mean())

# Define features and target variable
X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
y = df['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Comparison
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.4f}")

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100],  # Reduced grid size
    'max_depth': [None, 10, 20],  # Reduced grid size
    'min_samples_split': [2, 5]  # Reduced grid size
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters for Random Forest:", grid_search.best_params_)

# Evaluate the best model
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)

print("\nBest Random Forest Model Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred_best))

# Feature Importance Visualization
importances = best_rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
plt.close()
