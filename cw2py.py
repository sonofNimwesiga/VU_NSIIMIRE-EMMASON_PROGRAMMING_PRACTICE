#FLOWCHART VISUALISATION
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6.5), dpi=100)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Styling options
box_dark = dict(boxstyle="round,pad=0.6", facecolor="#1B365D", edgecolor="#0D1B2A", lw=1.5)
box_blue = dict(boxstyle="round,pad=0.6", facecolor="#2E5B88", edgecolor="#0D1B2A", lw=1.5)
box_green = dict(boxstyle="round,pad=0.6", facecolor="#1E7145", edgecolor="#0D1B2A", lw=1.5)
arrow_style = dict(arrowstyle="->", lw=2.5, color="#1B365D")

# Nodes definition
nodes = [
    (5, 13, "Raw Student Performance Dataset\n(10,000 Records)", box_dark),
    (5, 10.8, "Task 1: Data Cleaning & Preprocessing\n• Remove 127 Duplicate Records\n• Encode Extracurricular (Yes/No → 1/0)\n• Define Target: pass_status (1 if PI ≥ 50 else 0)", box_blue),
    (5, 8.2, "Task 2: Data Splitting & Feature Scaling\n• Stratified Train-Test Split (80% Train / 20% Test)\n• Standard Scaling (Z-score Normalization on X_train)", box_blue),
    (5, 5.6, "Task 3: Supervised Model Training\n• Baseline Model: Logistic Regression\n• Proposed Benchmark: Random Forest Classifier", box_blue),
    (5, 3.0, "Task 4: Model Evaluation & Predictions\n• Performance Metrics (Accuracy, Precision, Recall, F1)\n• Confusion Matrix Analysis & Error Diagnosis\n• Interactive / Manual Student Risk Tester", box_green),
    (5, 0.8, "Final Coursework Report & Deployment Ready", box_dark)
]

# Draw nodes
for x, y, text, box in nodes:
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=10, fontweight="bold", bbox=box, multialignment="left" if "•" in text else "center")

# Draw connecting arrows
arrows = [(12.1, 11.7), (9.9, 9.0), (7.4, 6.4), (4.8, 3.8), (2.1, 1.2)]
for y_start, y_end in arrows:
    ax.annotate("", xy=(5, y_end), xytext=(5, y_start), arrowprops=arrow_style)

plt.title("Machine Learning Workflow Flowchart\nStudent Performance Classification & Score Prediction", fontsize=14, fontweight="bold", color="#1B365D", pad=20)
plt.tight_layout()
plt.savefig("ML_Pipeline_Flowchart.png", dpi=300, bbox_inches="tight")
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Load and Inspect Raw Data
df = pd.read_csv(r"C:\Users\user\Desktop\Student_Performance.csv")

print("RAW DATA SUMMARY")
print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nData Types & Non-Null Counts:")
print(df.info())
print("\nFirst 50 Records:")
print(df.head(50))

#Data Profiling 
print("\nMissing Values Count:")
print(df.isnull().sum())

print(f"\nDuplicate Records Count: {df.duplicated().sum()}")

print("\nDescriptive Statistics:")
print(df.describe())

print("\nCategorical Distribution (Extracurricular Activities):")
print(df['Extracurricular Activities'].value_counts())

#Custom Cleaning & Transformation Function
def prepare_student_dataset(df):
    """
    Custom preprocessing function:
    1. Removes duplicate rows
    2. Standardizes column names
    3. Encodes categorical binary variables
    4. Constructs target binary class for classification tasks
    5. Scales numerical feature columns using StandardScaler
    """
    cleaned_df = df.copy()
    
    # 1. Remove duplicate entries
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 2. Standardize column headers
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # 3. Binary categorical encoding
    cleaned_df['extracurricular_binary'] = (cleaned_df['extracurricular_activities'] == 'Yes').astype(int)
    
    # 4. Feature Construction: Target for Classification (Pass/Fail threshold at 50)
    cleaned_df['pass_status'] = (cleaned_df['performance_index'] >= 50.0).astype(int)
    
    return cleaned_df

# Apply preprocessing function
prep_df = prepare_student_dataset(df)

# Feature Scaling
feature_cols = ['hours_studied', 'previous_scores', 'sleep_hours', 'sample_question_papers_practiced', 'extracurricular_binary']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(prep_df[feature_cols])
scaled_feature_df = pd.DataFrame(scaled_features, columns=[f"{col}_scaled" for col in feature_cols], index=prep_df.index)

# Merge scaled features back
final_df = pd.concat([prep_df, scaled_feature_df], axis=1)

# Save cleaned dataset
final_df.to_csv('Prepared_Student_Performance.csv', index=False)

#implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             r2_score, mean_squared_error, mean_absolute_error)

# Load prepared data
df = pd.read_csv('Prepared_Student_Performance.csv')

# Define features and classification target
feature_cols = ['hours_studied', 'previous_scores', 'sleep_hours', 'sample_question_papers_practiced', 'extracurricular_binary']
X = df[feature_cols]
y_class = df['pass_status']
y_reg = df['performance_index']

# Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.20, random_state=42, stratify=y_class
)

# Scale features for distance/gradient-sensitive models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#CUSTOM WRAPPER CLASS FOR MODEL EVALUATION
class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        
    def fit_and_evaluate(self, X_tr, y_tr, X_te, y_te):
        self.model.fit(X_tr, y_tr)
        predictions = self.model.predict(X_te)
        
        metrics = {
            'Accuracy': accuracy_score(y_te, predictions),
            'Precision': precision_score(y_te, predictions),
            'Recall': recall_score(y_te, predictions),
            'F1-Score': f1_score(y_te, predictions)
        }
        return metrics, predictions

# 1. Baseline Model: Logistic Regression
lr_evaluator = ModelEvaluator(LogisticRegression(random_state=42))
lr_metrics, y_pred_lr = lr_evaluator.fit_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

# 2. Proposed Model: Random Forest Classifier
rf_evaluator = ModelEvaluator(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
rf_metrics, y_pred_rf = rf_evaluator.fit_and_evaluate(X_train, y_train, X_test, y_test)

# Output Classification Results
print("CLASSIFICATION PERFORMANCE COMPARISON")
print(f"Logistic Regression (Baseline) -> Accuracy: {lr_metrics['Accuracy']:.4f}, Precision: {lr_metrics['Precision']:.4f}, Recall: {lr_metrics['Recall']:.4f}, F1: {lr_metrics['F1-Score']:.4f}")
print(f"Random Forest (Proposed)       -> Accuracy: {rf_metrics['Accuracy']:.4f}, Precision: {rf_metrics['Precision']:.4f}, Recall: {rf_metrics['Recall']:.4f}, F1: {rf_metrics['F1-Score']:.4f}")

# Plot Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Fail (<50)', 'Pass (>=50)'], yticklabels=['Fail (<50)', 'Pass (>=50)'])
axes[0].set_title('Baseline: Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Fail (<50)', 'Pass (>=50)'], yticklabels=['Fail (<50)', 'Pass (>=50)'])
axes[1].set_title('Proposed: Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()
#OPTIONAL REGRESSION EVALUATION (Predicting Continuous Performance Index)
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, y_reg, test_size=0.20, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(scaler.fit_transform(X_tr_r), y_tr_r)
y_pred_reg = lin_reg.predict(scaler.transform(X_te_r))

r2 = r2_score(y_te_r, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_te_r, y_pred_reg))
mae = mean_absolute_error(y_te_r, y_pred_reg)

print("\nDIRECT REGRESSION RESULTS (Linear Regression)")
print(f"R² Score: {r2:.4f}")
print(f"RMSE:     {rmse:.4f}")
print(f"MAE:      {mae:.4f}")



#TESTING MODEL
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# 1. LOAD DATA & TRAIN MODEL PIPELINE

# Load dataset
df = pd.read_csv('Prepared_Student_Performance.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Create binary features
df['extracurricular_binary'] = (df['extracurricular_activities'] == 'Yes').astype(int)
df['pass_status'] = (df['performance_index'] >= 50.0).astype(int)

# Separate features & target
feature_cols = ['hours_studied', 'previous_scores', 'sleep_hours', 'sample_question_papers_practiced', 'extracurricular_binary']
X = df[feature_cols]
y = df['pass_status']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)


# 2. MANUAL TESTING INTERACTIVE FUNCTION

def manual_student_test():
    print("\n" + "="*50)
    print("   STUDENT PERFORMANCE PREDICTION SYSTEM   ")
    print("="*50)
    
    try:
        # Prompt user for inputs
        hours_studied = float(input("Enter Hours Studied (1 - 9): "))
        previous_scores = float(input("Enter Previous Exam Score (40 - 99): "))
        sleep_hours = float(input("Enter Sleep Hours per night (4 - 9): "))
        sample_papers = float(input("Enter Sample Question Papers Practiced (0 - 9): "))
        extracurricular = input("Participates in Extracurricular Activities? (Yes / No): ").strip().lower()
        
        # Convert extracurricular to binary 0/1
        extra_binary = 1 if extracurricular in ['yes', 'y', '1'] else 0
        
        # Format input DataFrame matching training columns
        student_data = pd.DataFrame([{
            'hours_studied': hours_studied,
            'previous_scores': previous_scores,
            'sleep_hours': sleep_hours,
            'sample_question_papers_practiced': sample_papers,
            'extracurricular_binary': extra_binary
        }])
        
        # Apply fitted StandardScaler
        student_scaled = scaler.transform(student_data)
        
        # Generate prediction and confidence probability
        prediction = model.predict(student_scaled)[0]
        pass_prob = model.predict_proba(student_scaled)[0][1]
        
        # Display output
        print("-" * 50)
        print("                 PREDICTION RESULTS                 ")
        print("-" * 50)
        if prediction == 1:
            print(f" Status          : PASS [Pass Status = 1]")
        else:
            print(f" Status          : FAIL [Pass Status = 0]")
            
        print(f" Pass Probability: {pass_prob * 100:.2f}%")
        print(f" Fail Probability: {(1 - pass_prob) * 100:.2f}%")
        print("="*50 + "\n")
        
    except ValueError:
        print("\n Invalid input! Please enter numerical values for metrics.\n")


# 3. RUN INTERACTIVE TESTING LOOP

if __name__ == "__main__":
    while True:
        manual_student_test()
        again = input("Do you want to test another student? (yes/no): ").strip().lower()
        if again not in ['yes', 'y']:
            print("\nExiting Student Performance Tester. Goodbye!")
            break