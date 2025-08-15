import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
DATA_DIR = 'data'  # Update this path if your data is located elsewhere
OUTPUT_DIR = 'joai/exp11'  # Directory to save models

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

# Standardize features for TabNet
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# K-Fold Cross-validation
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
    
    X_train_fold, X_val_fold = X_scaled_df.loc[train_idx], X_scaled_df.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    
    print(f"  Train: {len(X_train_fold)} samples, Validation: {len(X_val_fold)} samples")
    
    # Define TabNet model
    tabnet_params = {
        'n_d': 8,  # Width of the decision prediction layer
        'n_a': 8,  # Width of the attention embedding
        'n_steps': 3,  # Number of steps in the architecture
        'gamma': 1.5,  # Scaling factor for attention
        'n_independent': 2,  # Number of independent GLU layers at each step
        'n_shared': 2,  # Number of shared GLU layers at each step
        'lambda_sparse': 1e-3,  # Sparsity regularization
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': {'lr': 2e-2},
        'scheduler_params': {"step_size": 10, "gamma": 0.9},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'mask_type': 'entmax',  # "sparsemax" or "entmax"
        'seed': 42,
        'verbose': 1
    }
    
    # Initialize and train model
    model = TabNetClassifier(**tabnet_params)
    model.fit(
        X_train=X_train_fold.values,
        y_train=y_train_fold.values,
        eval_set=[(X_val_fold.values, y_val_fold.values)],
        eval_name=['val'],
        eval_metric=['accuracy'],
        max_epochs=200,
        patience=20,
        batch_size=256,
        virtual_batch_size=128
    )
    
    # Store the model
    fold_models.append(model)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'tabnet_model_fold_{fold}.zip')
    model.save_model(model_path)
    print(f"  Model saved to {model_path}")
    
    # Make predictions
    val_preds = model.predict_proba(X_val_fold.values)
    oof_predictions[val_idx] = val_preds
    val_pred_class = np.argmax(val_preds, axis=1)
    
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
    
    # Feature importance for this fold
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'Feature Importance - Fold {fold + 1}')
    plt.tight_layout()
    #plt.savefig(os.path.join(OUTPUT_DIR, f'feature_importance_fold_{fold}.png'))
    plt.close()

# Calculate OOF metrics
oof_pred_class = np.argmax(oof_predictions, axis=1)
oof_accuracy = accuracy_score(y, oof_pred_class)
oof_f1 = f1_score(y, oof_pred_class, average='weighted')

print("\nOut-of-fold prediction results:")
print(f"OOF Accuracy: {oof_accuracy:.4f}")
print(f"OOF F1 Score: {oof_f1:.4f}")

# Save OOF predictions with probabilities
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

# Visualize decision boundaries with average prediction from all folds
plt.figure(figsize=(12, 10))

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X['MQ8'].min() - 50, X['MQ8'].max() + 50
y_min, y_max = X['MQ5'].min() - 50, X['MQ5'].max() + 50
xx, yy = np.meshgrid(np.arange(x_min, x_max, 5), np.arange(y_min, y_max, 5))
mesh_points = np.c_[xx.ravel(), yy.ravel()]

# Create a DataFrame with default values for other features
mesh_df = pd.DataFrame()
mesh_df['MQ8'] = xx.ravel()
mesh_df['MQ5'] = yy.ravel()
# Use median values for other features
mesh_df['temp_min'] = np.full(xx.ravel().shape[0], X['temp_min'].median())
mesh_df['temp_max'] = np.full(xx.ravel().shape[0], X['temp_max'].median())
mesh_df['temp_unit_encoded'] = np.full(xx.ravel().shape[0], X['temp_unit_encoded'].median())
mesh_df['has_coordinates_encoded'] = np.full(xx.ravel().shape[0], X['has_coordinates_encoded'].median())

# Scale mesh points using the same scaler
mesh_points_scaled = scaler.transform(mesh_df[X.columns].values)

# Get predictions for all points in the mesh from all models
mesh_preds = np.zeros((mesh_points_scaled.shape[0], len(label_encoder.classes_)))
for model in fold_models:
    mesh_preds += model.predict_proba(mesh_points_scaled)
mesh_preds /= len(fold_models)  # Average predictions

Z = np.argmax(mesh_preds, axis=1)
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot training points
for i, gas_type in enumerate(label_encoder.classes_):
    idx = train_df[train_df['Gas'] == gas_type].index
    plt.scatter(train_df.loc[idx, 'MQ8'], train_df.loc[idx, 'MQ5'], 
                c=f'C{i}', label=gas_type, edgecolors='k', alpha=0.7)

plt.title('Decision Boundaries with MQ8 and MQ5 (Average of All Folds)')
plt.xlabel('MQ8')
plt.ylabel('MQ5')
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
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Average predictions from all fold models
    test_preds_ensemble = np.zeros((len(X_test), len(label_encoder.classes_)))
    for model in fold_models:
        test_preds_ensemble += model.predict_proba(X_test_scaled)
    test_preds_ensemble /= len(fold_models)
    test_pred_class_ensemble = np.argmax(test_preds_ensemble, axis=1)
    
    # Convert numerical predictions back to original labels
    test_pred_labels_ensemble = label_encoder.inverse_transform(test_pred_class_ensemble)
    
    # Create submission dataframe
    submission = pd.DataFrame()
    submission['index'] = test_df.index
    submission['Gas'] = test_pred_labels_ensemble
    
    # Save submission file
    submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print("Submission file saved to submission.csv")
    
    # Save test predictions with probabilities
    test_probs_df = pd.DataFrame(test_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit', 'has_coordinates']].copy())
    # Add probabilities
    for i, gas_type in enumerate(label_encoder.classes_):
        test_probs_df[f'prob_{gas_type}'] = test_preds_ensemble[:, i]
    
    test_probs_df['predicted_label'] = test_pred_labels_ensemble
    test_probs_df.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv'), index=False)
    print(f"Test predictions with probabilities saved to {os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv')}")

print("Done!")
