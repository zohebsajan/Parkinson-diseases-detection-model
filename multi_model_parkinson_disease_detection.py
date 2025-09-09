"""
Parkinson's Disease Detection System
===================================
A comprehensive machine learning system to detect Parkinson's disease using voice measurements.

Features:
- Multiple ML algorithms comparison (SVM, Random Forest, Logistic Regression)
- Cross-validation for robust evaluation
- Feature importance analysis
- Comprehensive data visualization
- Interactive prediction interface
- Model persistence for deployment

Author: Enhanced ML Project
Date: September 2025
"""

# Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PARKINSON'S DISEASE DETECTION SYSTEM")
print("="*60)

# Data Loading and Exploration
print("\n1. LOADING AND EXPLORING DATA")
print("-"*40)

# Loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('archive\\parkinsons.csv')

# Basic information about the dataset
print(f"Dataset shape: {parkinsons_data.shape}")
print(f"Number of features: {parkinsons_data.shape[1] - 2}")  # Excluding 'name' and 'status'

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(parkinsons_data.head())

# Dataset information
print("\nDataset Information:")
parkinsons_data.info()

# Check for missing values
print("\nMissing values in each column:")
missing_values = parkinsons_data.isnull().sum()
if missing_values.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing_values[missing_values > 0])

# Target variable distribution
print(f"\nTarget Variable Distribution:")
status_counts = parkinsons_data['status'].value_counts()
print(f"Healthy (0): {status_counts[0]} samples")
print(f"Parkinson's (1): {status_counts[1]} samples")
print(f"Class balance ratio: {status_counts[1]/status_counts[0]:.2f}")

# Statistical summary
print("\nStatistical Summary:")
print(parkinsons_data.describe())

# Group statistics by target variable
print("\nMean values by diagnosis:")
numeric_columns = parkinsons_data.select_dtypes(include=np.number).columns
grouped_stats = parkinsons_data.groupby('status')[numeric_columns].mean()
print(grouped_stats)

# Data Visualization
print("\n2. DATA VISUALIZATION")
print("-"*40)

# Set up the plotting style
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(20, 15))

# 1. Target distribution pie chart
plt.subplot(2, 3, 1)
labels = ['Healthy', "Parkinson's"]
sizes = [status_counts[0], status_counts[1]]
colors = ['lightblue', 'lightcoral']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribution of Target Variable')

# 2. Feature correlation heatmap (top features)
plt.subplot(2, 3, 2)
# Select top 15 features for better visualization
correlation_matrix = parkinsons_data.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix.iloc[:15, :15], mask=mask[:15, :15], 
            cmap='coolwarm', center=0, annot=False)
plt.title('Feature Correlation Matrix (Top 15)')

# 3. Box plot for selected features
plt.subplot(2, 3, 3)
selected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)']
parkinsons_melted = pd.melt(parkinsons_data, id_vars=['status'], 
                           value_vars=selected_features, 
                           var_name='feature', value_name='value')
sns.boxplot(data=parkinsons_melted, x='feature', y='value', hue='status')
plt.xticks(rotation=45)
plt.title('Feature Distribution by Diagnosis')
plt.legend(['Healthy', "Parkinson's"])

plt.tight_layout()
plt.show()

# Data Preprocessing
print("\n3. DATA PREPROCESSING")
print("-"*40)

# Separating features and target
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

print(f"Features shape: {X.shape}")
print(f"Target shape: {Y.shape}")
print(f"Feature names: {list(X.columns[:5])}... (showing first 5)")

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state=42, stratify=Y)

print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Data standardization completed")

# Model Training and Evaluation
print("\n4. MODEL TRAINING AND EVALUATION")
print("-"*40)

# Initialize models
models = {
    'Support Vector Machine': svm.SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Store results
results = {}
model_objects = {}

print("Training and evaluating models...")
print()

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, Y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_accuracy = accuracy_score(Y_train, y_train_pred)
    test_accuracy = accuracy_score(Y_test, y_test_pred)
    precision = precision_score(Y_test, y_test_pred)
    recall = recall_score(Y_test, y_test_pred)
    f1 = f1_score(Y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, Y_train, cv=5, scoring='accuracy')
    
    # Store results
    results[name] = {
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Predictions': y_test_pred
    }
    
    model_objects[name] = model
    
    print(f"✓ {name} completed")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()

# Results Summary
print("5. RESULTS SUMMARY")
print("-"*40)

results_df = pd.DataFrame(results).T
print("Performance Comparison:")
print(results_df[['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean']].round(4))

# Find best model
best_model_name = results_df['Test Accuracy'].idxmax()
best_model = model_objects[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"Best Test Accuracy: {results_df.loc[best_model_name, 'Test Accuracy']:.4f}")

# Detailed classification report for best model
print(f"\nDetailed Classification Report - {best_model_name}:")
print(classification_report(Y_test, results[best_model_name]['Predictions']))

# Visualization of Results
print("\n6. RESULTS VISUALIZATION")
print("-"*40)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model comparison bar plot
metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df.index))
width = 0.2

for i, metric in enumerate(metrics):
    ax1.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)

ax1.set_xlabel('Models')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(results_df.index, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Confusion matrix for best model
cm = confusion_matrix(Y_test, results[best_model_name]['Predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Healthy', "Parkinson's"],
            yticklabels=['Healthy', "Parkinson's"])
ax2.set_title(f'Confusion Matrix - {best_model_name}')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

# 3. Feature importance (for Random Forest)
if 'Random Forest' in model_objects:
    rf_model = model_objects['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    ax3.barh(range(len(feature_importance)), feature_importance['importance'])
    ax3.set_yticks(range(len(feature_importance)))
    ax3.set_yticklabels(feature_importance['feature'])
    ax3.set_title('Top 10 Feature Importance (Random Forest)')
    ax3.set_xlabel('Importance')

# 4. Cross-validation scores
cv_means = [results[model]['CV Mean'] for model in results.keys()]
cv_stds = [results[model]['CV Std'] for model in results.keys()]
model_names = list(results.keys())

ax4.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon'])
ax4.set_title('Cross-Validation Scores')
ax4.set_ylabel('Accuracy')
ax4.set_xticklabels(model_names, rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Model Persistence
print("\n7. MODEL PERSISTENCE")
print("-"*40)

# Save the best model and scaler
joblib.dump(best_model, 'best_parkinsons_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print(f"✓ Best model ({best_model_name}) saved as 'best_parkinsons_model.pkl'")
print("✓ Feature scaler saved as 'feature_scaler.pkl'")

# Interactive Prediction System
print("\n8. INTERACTIVE PREDICTION SYSTEM")
print("-"*40)

def predict_parkinsons(model, scaler, input_data, patient_name):
    """
    Make prediction for a new patient
    """
    # Convert to numpy array and reshape
    input_array = np.asarray(input_data).reshape(1, -1)
    
    # Standardize the input
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled) if hasattr(model, 'predict_proba') else None
    
    # Display results
    print(f"\n{'='*50}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"Patient Name: {patient_name}")
    
    if prediction[0] == 0:
        result = "HEALTHY - No indication of Parkinson's Disease"
        print(f"Diagnosis: {result}")
    else:
        result = "PARKINSON'S DISEASE DETECTED"
        print(f"Diagnosis: {result}")
    
    if prediction_proba is not None:
        confidence = max(prediction_proba[0]) * 100
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probability - Healthy: {prediction_proba[0][0]:.3f}")
        print(f"Probability - Parkinson's: {prediction_proba[0][1]:.3f}")
    
    print(f"{'='*50}")
    return prediction[0], result

# Example prediction with sample data
print("\nExample Prediction:")
print("Using sample data from the dataset...")

# Use a sample from the test set
sample_idx = 0
sample_data = X_test.iloc[sample_idx].values
actual_diagnosis = Y_test.iloc[sample_idx]
sample_name = f"Sample Patient {sample_idx + 1}"

predicted_class, diagnosis = predict_parkinsons(best_model, scaler, sample_data, sample_name)
actual_label = "Parkinson's Disease" if actual_diagnosis == 1 else "Healthy"
prediction_status = "✓ Yes" if predicted_class == actual_diagnosis else "✗ No"
print(f"Actual Diagnosis: {actual_label}")
print(f"Prediction Correct: {prediction_status}")

# Interactive input section
print(f"\n{'='*60}")
print("MANUAL PREDICTION INPUT")
print(f"{'='*60}")
print("You can now input patient data for real-time prediction.")
print("Required features (22 values):", list(X.columns))
print("\nNote: Please ensure you have all 22 feature values in the correct order.")
print("For demonstration purposes, you can use sample data from the dataset.")

try:
    # Get patient name
    patient_name = input("\nEnter patient's name: ")
    
    print("\nEnter the 22 feature values (comma-separated):")
    print("Example format: 119.992,157.302,74.997,0.00784,0.00007,...")
    
    # Get input data
    user_input = input("Patient data: ")
    
    if user_input.strip():
        # Parse input data
        input_values = [float(x.strip()) for x in user_input.split(',')]
        
        if len(input_values) != len(X.columns):
            print(f"Error: Expected {len(X.columns)} values, got {len(input_values)}")
        else:
            # Make prediction
            predicted_class, diagnosis = predict_parkinsons(best_model, scaler, input_values, patient_name)
            
    else:
        print("No input provided. Using sample data instead.")
        
except (ValueError, EOFError, KeyboardInterrupt):
    print("Input cancelled or invalid. Using sample data for demonstration.")

# Summary
print(f"\n{'='*60}")
print("PROJECT SUMMARY")
print(f"{'='*60}")
print(f"✓ Dataset: {parkinsons_data.shape[0]} samples with {X.shape[1]} features")
print(f"✓ Models Trained: {len(models)}")
print(f"✓ Best Model: {best_model_name}")
print(f"✓ Best Accuracy: {results_df.loc[best_model_name, 'Test Accuracy']:.4f}")
print(f"✓ Cross-validation: {results_df.loc[best_model_name, 'CV Mean']:.4f} ± {results_df.loc[best_model_name, 'CV Std']:.4f}")
print(f"✓ Model saved for deployment")
print(f"✓ Interactive prediction system ready")

print("\nKey Features:")
print("• Comprehensive data analysis and visualization")
print("• Multiple ML algorithms comparison")
print("• Cross-validation for robust evaluation")
print("• Feature importance analysis")
print("• Model persistence for deployment")
print("• Interactive prediction interface")
print("• Professional documentation")

print(f"\n{'='*60}")
print("PROJECT COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")