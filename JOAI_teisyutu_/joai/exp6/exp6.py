import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
DATA_DIR = 'data'  # Update this path if your data is located elsewhere
OUTPUT_DIR = 'joai/exp6'  # Directory to save models

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the data
print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_folds_temp.csv'))
test_df = None

# Check if test.csv exists
test_path = os.path.join(DATA_DIR, 'test_temp.csv')
if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")

print(f"Training data shape: {train_df.shape}")
print(f"Target distribution: \n{train_df['Gas'].value_counts()}")

# Make sure MQ8, MQ5, temp_min and temp_max are treated as numeric
train_df['MQ8'] = pd.to_numeric(train_df['MQ8'], errors='coerce')
train_df['MQ5'] = pd.to_numeric(train_df['MQ5'], errors='coerce')
train_df['temp_min'] = pd.to_numeric(train_df['temp_min'], errors='coerce')
train_df['temp_max'] = pd.to_numeric(train_df['temp_max'], errors='coerce')

# Check for missing values after conversion
print(f"Missing values in MQ8: {train_df['MQ8'].isna().sum()}")
print(f"Missing values in MQ5: {train_df['MQ5'].isna().sum()}")
print(f"Missing values in temp_min: {train_df['temp_min'].isna().sum()}")
print(f"Missing values in temp_max: {train_df['temp_max'].isna().sum()}")

# Fill missing values with median if any
if train_df['MQ8'].isna().sum() > 0:
    train_df['MQ8'] = train_df['MQ8'].fillna(train_df['MQ8'].median())
if train_df['MQ5'].isna().sum() > 0:
    train_df['MQ5'] = train_df['MQ5'].fillna(train_df['MQ5'].median())
if train_df['temp_min'].isna().sum() > 0:
    train_df['temp_min'] = train_df['temp_min'].fillna(train_df['temp_min'].median())
if train_df['temp_max'].isna().sum() > 0:
    train_df['temp_max'] = train_df['temp_max'].fillna(train_df['temp_max'].median())

# Encode categorical features: temp_unit and has_coordinates
label_encoder_unit = LabelEncoder()
label_encoder_coordinates = LabelEncoder()

train_df['temp_unit_encoded'] = label_encoder_unit.fit_transform(train_df['temp_unit'])
train_df['has_coordinates_encoded'] = label_encoder_coordinates.fit_transform(train_df['has_coordinates'])

# Explore the relationship between temperature features and Gas
plt.figure(figsize=(10, 8))
sns.scatterplot(x='temp_min', y='temp_max', hue='Gas', data=train_df)
plt.title('Temperature Min vs Max by Gas Type')
plt.tight_layout()
#plt.savefig(os.path.join(OUTPUT_DIR, 'temp_min_max_scatter.png'))
plt.close()

# Explore the relationship between MQ8, MQ5 and Gas
plt.figure(figsize=(10, 8))
sns.scatterplot(x='MQ8', y='MQ5', hue='Gas', data=train_df)
plt.title('MQ8 vs MQ5 by Gas Type')
plt.tight_layout()
#plt.savefig(os.path.join(OUTPUT_DIR, 'mq8_mq5_scatter.png'))
plt.close()

# Encode the target variable
label_encoder = LabelEncoder()
train_df['Gas_encoded'] = label_encoder.fit_transform(train_df['Gas'])

# Map encoded values back to original labels for reference
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label mapping:", label_mapping)

# Define the features and target - now including temperature features
X = train_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit_encoded', 'has_coordinates_encoded']]
y = train_df['Gas_encoded']

# Define Random Forest parameters
rf_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': 42,
    'class_weight': 'balanced'
}

# K-Fold Cross-validation using pre-defined folds in train_folds_temp.csv
print("\nPerforming K-Fold Cross-validation...")
n_folds = train_df['fold'].nunique()
cv_scores = {'accuracy': [], 'f1': []}
oof_predictions = np.zeros((len(train_df), len(label_encoder.classes_)))
fold_models = []

for fold in range(n_folds):
    print(f"Training fold {fold + 1}/{n_folds}")
    
    # Split data using the fold column
    train_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index
    
    X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    
    print(f"  Train: {len(X_train_fold)} samples, Validation: {len(X_val_fold)} samples")
    
    # Train Random Forest model
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train_fold, y_train_fold)
    
    # Store the model
    fold_models.append(model)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'rf_model_fold_{fold}.joblib')
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")
    
    # Make predictions
    val_pred_class = model.predict(X_val_fold)
    val_preds = model.predict_proba(X_val_fold)
    oof_predictions[val_idx] = val_preds
    
    # Calculate metrics
    accuracy = accuracy_score(y_val_fold, val_pred_class)
    f1 = f1_score(y_val_fold, val_pred_class, average='weighted')
    
    cv_scores['accuracy'].append(accuracy)
    cv_scores['f1'].append(f1)
    
    print(f"  Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Plot confusion matrix for this fold
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_val_fold, val_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold + 1}')
    plt.tight_layout()
    #plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_fold_{fold}.png'))
    plt.close()

# Calculate OOF metrics
oof_pred_class = np.argmax(oof_predictions, axis=1)
oof_accuracy = accuracy_score(y, oof_pred_class)
oof_f1 = f1_score(y, oof_pred_class, average='weighted')

print("\nOut-of-fold prediction results:")
print(f"OOF Accuracy: {oof_accuracy:.4f}")
print(f"OOF F1 Score: {oof_f1:.4f}")

# Save OOF predictions with probabilities to CSV
oof_df = train_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit', 'has_coordinates', 'Gas', 'Gas_encoded', 'fold']].copy()
for i, gas_type in enumerate(label_encoder.classes_):
    oof_df[f'prob_{gas_type}'] = oof_predictions[:, i]
oof_df['predicted_class'] = oof_pred_class
oof_df['predicted_gas'] = label_encoder.inverse_transform(oof_pred_class)
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_predictions_probs.csv'), index=False)
print(f"OOF predictions with probabilities saved to {os.path.join(OUTPUT_DIR, 'oof_predictions_probs.csv')}")

print("\nCross-validation results:")
print(f"Mean Accuracy: {np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}")
print(f"Mean F1 Score: {np.mean(cv_scores['f1']):.4f} ± {np.std(cv_scores['f1']):.4f}")

print("\nClassification Report (OOF predictions):")
print(classification_report(y, oof_pred_class, target_names=label_encoder.classes_))

# Plot overall confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y, oof_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Overall Confusion Matrix (OOF Predictions)')
plt.tight_layout()
#plt.savefig(os.path.join(OUTPUT_DIR, 'overall_confusion_matrix.png'))
plt.close()

# Feature importance (average across folds)
plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame()
feature_names = ['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit_encoded', 'has_coordinates_encoded']

for i, model in enumerate(fold_models):
    fold_importance = pd.DataFrame()
    fold_importance['feature'] = feature_names
    fold_importance['importance'] = model.feature_importances_
    fold_importance['fold'] = i + 1
    importance_df = pd.concat([importance_df, fold_importance], axis=0)

avg_importance = importance_df.groupby('feature')['importance'].mean().reset_index()
avg_importance = avg_importance.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=avg_importance)
plt.title('Average Feature Importance Across Folds')
plt.tight_layout()
#plt.savefig(os.path.join(OUTPUT_DIR, 'average_feature_importance.png'))
plt.close()

# For visualization, we'll create a 2D plot with the most important features
# Determine the top 2 features
top_features = avg_importance['feature'].head(2).tolist()

# Visualize decision boundaries with average prediction from all folds
plt.figure(figsize=(12, 10))

# Create a mesh grid for plotting decision boundaries using top 2 features
x_min, x_max = train_df[top_features[0]].min() - 50, train_df[top_features[0]].max() + 50
y_min, y_max = train_df[top_features[1]].min() - 50, train_df[top_features[1]].max() + 50
xx, yy = np.meshgrid(np.arange(x_min, x_max, 5), np.arange(y_min, y_max, 5))

# Prepare mesh points with default values for other features
mesh_data = {}
for feat in feature_names:
    if feat not in top_features:
        # Use median values for features not in visualization
        mesh_data[feat] = np.full(xx.ravel().shape[0], train_df[feat].median())

# Add the top 2 features that we're visualizing
mesh_data[top_features[0]] = xx.ravel()
mesh_data[top_features[1]] = yy.ravel()

# Convert to DataFrame for consistent ordering of features
mesh_df = pd.DataFrame(mesh_data)
mesh_points = mesh_df[feature_names].values

# Get predictions for all points in the mesh from all models
mesh_preds = np.zeros((mesh_points.shape[0], len(label_encoder.classes_)))
for model in fold_models:
    mesh_preds += model.predict_proba(mesh_points)
mesh_preds /= len(fold_models)  # Average predictions

Z = np.argmax(mesh_preds, axis=1)
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot training points
for i, gas_type in enumerate(label_encoder.classes_):
    idx = train_df[train_df['Gas'] == gas_type].index
    plt.scatter(train_df.loc[idx, top_features[0]], train_df.loc[idx, top_features[1]], 
                c=f'C{i}', label=gas_type, edgecolors='k', alpha=0.7)

plt.title(f'Decision Boundaries with {top_features[0]} and {top_features[1]} (Average of All Folds)')
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.legend()
#plt.savefig(os.path.join(OUTPUT_DIR, 'decision_boundaries.png'))
plt.close()

# Make predictions for test data if available
if test_df is not None:
    # Process test data similar to training data
    test_df['MQ8'] = pd.to_numeric(test_df['MQ8'], errors='coerce')
    test_df['MQ5'] = pd.to_numeric(test_df['MQ5'], errors='coerce')
    test_df['temp_min'] = pd.to_numeric(test_df['temp_min'], errors='coerce')
    test_df['temp_max'] = pd.to_numeric(test_df['temp_max'], errors='coerce')
    
    # Fill missing values with training median if any
    if test_df['MQ8'].isna().sum() > 0:
        test_df['MQ8'] = test_df['MQ8'].fillna(train_df['MQ8'].median())
    if test_df['MQ5'].isna().sum() > 0:
        test_df['MQ5'] = test_df['MQ5'].fillna(train_df['MQ5'].median())
    if test_df['temp_min'].isna().sum() > 0:
        test_df['temp_min'] = test_df['temp_min'].fillna(train_df['temp_min'].median())
    if test_df['temp_max'].isna().sum() > 0:
        test_df['temp_max'] = test_df['temp_max'].fillna(train_df['temp_max'].median())
    
    # Encode categorical features using training fit
    test_df['temp_unit_encoded'] = label_encoder_unit.transform(test_df['temp_unit'])
    test_df['has_coordinates_encoded'] = label_encoder_coordinates.transform(test_df['has_coordinates'])
    
    # Extract features
    X_test = test_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit_encoded', 'has_coordinates_encoded']]
    
    # Average predictions from all fold models
    test_preds_ensemble = np.zeros((len(X_test), len(label_encoder.classes_)))
    for model in fold_models:
        test_preds_ensemble += model.predict_proba(X_test)
    test_preds_ensemble /= len(fold_models)  # Average predictions
    test_pred_class_ensemble = np.argmax(test_preds_ensemble, axis=1)
    
    # Convert numerical predictions back to original labels
    test_pred_labels_ensemble = label_encoder.inverse_transform(test_pred_class_ensemble)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'index': test_df.index,
        'Gas': test_pred_labels_ensemble
    })
    
    # Save submission
    submission_df.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print("Submission file created.")
    
    # Save test predictions with probabilities to CSV
    test_probs_df = pd.DataFrame(test_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit', 'has_coordinates']].copy())
    # Add probabilities
    for i, gas_type in enumerate(label_encoder.classes_):
        test_probs_df[f'prob_{gas_type}'] = test_preds_ensemble[:, i]
    
    test_probs_df['predicted_label'] = test_pred_labels_ensemble
    test_probs_df.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv'), index=False)
    print(f"Test predictions with probabilities saved to {os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv')}")

print("Experiment completed successfully!")
