import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
import shap
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/content/drive/MyDrive/ML AI/Project/thyroidcancer.csv'  # Update path as needed

def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit()

df = load_dataset(file_path)

# Handle missing values
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype == 'object':
                mode_values = df[col].mode()
                if not mode_values.empty:
                    df[col] = df[col].fillna(mode_values[0])
            else:
                df[col] = df[col].fillna(df[col].median())

handle_missing_values(df)

# Define features and target variable
X = df.drop(columns=['death', 'recurred', 'Success', 'index', 'patientID.blind', 'TumorID.blind'])
y = df['Success']

# Preprocessing pipelines
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Preprocess features
X_processed = preprocessor.fit_transform(X)
X_processed = pd.DataFrame(X_processed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_xgb_model.predict(X_test)
y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_xgb_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - XGBoost Best Model")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Best Model")
plt.legend(loc="lower right")
plt.show()

# SHAP Analysis
feature_names = numerical_features.tolist() + preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features).tolist()

explainer = shap.Explainer(best_xgb_model, X_train)
shap_values = explainer(X_test)

shap_values_with_names = shap.Explanation(
    values=shap_values.values,
    base_values=shap_values.base_values,
    feature_names=feature_names
)

shap.plots.waterfall(shap_values_with_names[0], max_display=10)
plt.savefig("shap_waterfall_plot.png")
print("Waterfall SHAP plot saved as shap_waterfall_plot.png")
