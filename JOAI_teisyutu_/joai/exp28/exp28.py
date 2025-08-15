import os
import numpy as np
import pandas as pd
from scipy.special import softmax, logit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
DATA_DIR = 'data'
# Experiment directories
EXP_DIRS = [
    'joai/exp0',
    'joai/exp1',
    'joai/exp3',
    'joai/exp4',
    'joai/exp5',
    'joai/exp6',
    'joai/exp7',
    'joai/exp8',
    'joai/exp11',
    'joai/exp13',
    'joai/exp14',
    'joai/exp16',
    'joai/exp21',
    'joai/exp22',
    'joai/exp24',
    'joai/exp26',
    'joai/exp27',
]

OUTPUT_DIR = 'joai/exp28'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading experiment data...")

# Ensemble OOF predictions
all_oof_dfs = []
for exp_dir in EXP_DIRS:
    exp_name = os.path.basename(exp_dir)
    oof_path = os.path.join(exp_dir, 'oof_predictions_probs.csv')
    if os.path.exists(oof_path):
        oof_df = pd.read_csv(oof_path)
        all_oof_dfs.append((exp_name, oof_df))
        print(f"Loaded OOF predictions from {exp_name}")
    else:
        print(f"Warning: OOF predictions file not found for {exp_name}")

# Ensemble test predictions
all_test_dfs = []
for exp_dir in EXP_DIRS:
    exp_name = os.path.basename(exp_dir)
    test_path = os.path.join(exp_dir, 'test_predictions_probs.csv')
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        all_test_dfs.append((exp_name, test_df))
        print(f"Loaded test predictions from {exp_name}")
    else:
        print(f"Warning: Test predictions file not found for {exp_name}")

# Function to ensemble predictions by averaging in logit space
def ensemble_logit_average(pred_dfs, prob_cols):
    # Extract the first dataframe to get the structure
    base_df = pred_dfs[0][1].copy()
    
    # Initialize array to store logits from all models
    all_logits = []
    
    # Convert probabilities to logits for each model
    for _, df in pred_dfs:
        probs = df[prob_cols].values
        # Apply small epsilon to avoid log(0)
        probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        logits = logit(probs)
        all_logits.append(logits)
    
    # Average the logits
    avg_logits = np.mean(all_logits, axis=0)
    
    # Convert back to probabilities
    avg_probs = softmax(avg_logits, axis=1)
    
    # Update the base dataframe with new probabilities
    for i, col in enumerate(prob_cols):
        base_df[col] = avg_probs[:, i]
    
    # Update predicted class
    base_df['predicted_class'] = np.argmax(avg_probs, axis=1)
    
    if 'Gas' in base_df.columns and 'Gas_encoded' in base_df.columns:
        # Map back to original gas labels
        gas_mapping = dict(zip(base_df['Gas_encoded'].unique(), base_df['Gas'].unique()))
        base_df['predicted_gas'] = base_df['predicted_class'].map(gas_mapping)
    elif 'predicted_label' in base_df.columns:
        gas_types = [col.replace('prob_', '') for col in prob_cols]
        base_df['predicted_label'] = [gas_types[i] for i in base_df['predicted_class']]
    
    return base_df

# Function to create stacked features from model predictions
def create_stacked_features(pred_dfs, prob_cols):
    # Extract the first dataframe to get the structure and metadata columns
    base_df = pred_dfs[0][1].copy()
    
    # Get metadata columns (non-probability columns)
    meta_cols = [col for col in base_df.columns if not col.startswith('prob_')]
    meta_data = base_df[meta_cols].copy()
    
    # Create stacked features dataframe
    stacked_features = pd.DataFrame(index=base_df.index)
    
    # Add model predictions as features
    for model_name, df in pred_dfs:
        for col in prob_cols:
            # Add each model's probability prediction as a feature
            feature_name = f"{model_name}_{col}"
            stacked_features[feature_name] = df[col].values
    
    # Combine with metadata
    result_df = pd.concat([meta_data, stacked_features], axis=1)
    
    return result_df

# Function for stacking with LightGBM - IMPROVED VERSION for exp28
def stacking_lightgbm(oof_dfs, test_dfs, prob_cols):
    # Create stacked features for OOF and test predictions
    oof_stacked = create_stacked_features(oof_dfs, prob_cols)
    test_stacked = create_stacked_features(test_dfs, prob_cols)
    
    # Load the train_folds_temp.csv to get correct labels and original features
    train_folds_temp_path = os.path.join(DATA_DIR, 'train_folds_temp.csv')
    if not os.path.exists(train_folds_temp_path):
        print(f"Error: {train_folds_temp_path} not found.")
        return None, None
    
    train_folds_temp_df = pd.read_csv(train_folds_temp_path)
    
    # Get the image file paths from OOF if available 
    if 'image_path_uuid' in oof_stacked.columns:
        image_paths = oof_stacked['image_path_uuid']
    elif 'image_path' in oof_stacked.columns:
        image_paths = oof_stacked['image_path']
    else:
        print("Error: Cannot find image path column in OOF data.")
        return None, None
    
    # Map the correct labels from train_folds_temp to oof_stacked
    # Create a mapping dictionary for Gas labels
    gas_mapping = {
        'NoGas': 0,
        'Perfume': 1,
        'Smoke': 2,
        'Mixture': 3
    }
    
    # Add Gas and Gas_encoded columns to oof_stacked from train_folds_temp
    # We need to merge based on image paths
    file_names = [os.path.basename(path) if isinstance(path, str) else path for path in image_paths]
    oof_stacked['file_name'] = file_names
    
    # If train_folds_temp has the full path, extract just the filename
    if 'image_path_uuid' in train_folds_temp_df.columns:
        train_folds_temp_df['file_name'] = train_folds_temp_df['image_path_uuid'].apply(
            lambda x: os.path.basename(x) if isinstance(x, str) else x
        )
    
    # Merge to get the correct labels and original features
    oof_stacked = pd.merge(oof_stacked, 
                          train_folds_temp_df[['file_name', 'Gas', 'fold']], 
                          on='file_name', 
                          how='left',
                          suffixes=('_orig', ''))
        
    # Create Gas_encoded column if it doesn't exist
    if 'Gas_encoded' not in oof_stacked.columns:
        oof_stacked['Gas_encoded'] = oof_stacked['Gas'].map(gas_mapping)
    
    # Check if we have the target column now
    if 'Gas_encoded' not in oof_stacked.columns or oof_stacked['Gas_encoded'].isnull().any():
        print("Warning: Some target labels are missing after merging with train_folds_temp.")
        # Fill missing values with a default class if needed
        oof_stacked['Gas_encoded'] = oof_stacked['Gas_encoded'].fillna(0).astype(int)
    
    # Process test data
    test_df_path = os.path.join(DATA_DIR, 'test_temp.csv')
    if os.path.exists(test_df_path):
        test_df = pd.read_csv(test_df_path)
                
        # Get file names for test data
        if 'image_path_uuid' in test_stacked.columns:
            test_image_paths = test_stacked['image_path_uuid']
        elif 'image_path' in test_stacked.columns:
            test_image_paths = test_stacked['image_path']
        else:
            print("Warning: Cannot find image path column in test data.")
            test_image_paths = []
            
        if len(test_image_paths) > 0:
            test_file_names = [os.path.basename(path) if isinstance(path, str) else path for path in test_image_paths]
            test_stacked['file_name'] = test_file_names
            
            # If test_df has the full path, extract just the filename
            if 'image_path_uuid' in test_df.columns:
                test_df['file_name'] = test_df['image_path_uuid'].apply(
                    lambda x: os.path.basename(x) if isinstance(x, str) else x
                )
                
            # Merge to get original features
            test_stacked = pd.merge(test_stacked,
                                   test_df[['file_name']],
                                   on='file_name',
                                   how='left',
                                   suffixes=('_orig', ''))
    
    # Extract features for training - Now including both model predictions and original features
    # First, get model prediction features
    model_feature_cols = [col for col in oof_stacked.columns 
                          if any(exp in col for exp in [f"exp{i}" for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 21, 24, 26, 27]]) 
                          and col.startswith('exp')]
            
    # Combine all feature columns
    feature_cols = model_feature_cols
    
    print(f"Using {len(feature_cols)} features for stacking: {len(model_feature_cols)} model predictions + {len(feature_cols) - len(model_feature_cols)} original features")
    
    X_train = oof_stacked[feature_cols]
    y_train = oof_stacked['Gas_encoded']
    X_test = test_stacked[feature_cols]
    
    num_classes = len(prob_cols)
    oof_preds = np.zeros((len(X_train), num_classes))
    
    # Unlike exp10, we will store test predictions for each fold separately
    # Initialize a dictionary to store test predictions for each fold
    fold_test_preds = {}
    
    # LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Using the folds from train_folds_temp.csv instead of creating new splits
    n_folds = oof_stacked['fold'].nunique()
    print(f"Training using {n_folds} folds from train_folds_temp.csv")
    
    for fold in range(n_folds):
        print(f"Training fold {fold+1}/{n_folds}")
        
        # Get train and validation indices based on the fold column
        train_idx = oof_stacked[oof_stacked['fold'] != fold].index
        val_idx = oof_stacked[oof_stacked['fold'] == fold].index
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        print(f"  Train: {len(X_tr)} samples, Validation: {len(X_val)} samples")
        
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        # Train model - Using custom F1 evaluation function
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # OOF predictions
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Calculate fold-specific metrics
        val_pred_class = np.argmax(oof_preds[val_idx], axis=1)
        fold_accuracy = accuracy_score(y_val, val_pred_class)
        fold_f1 = f1_score(y_val, val_pred_class, average='weighted')
        print(f"  Fold {fold + 1} - Accuracy: {fold_accuracy:.4f}, F1 Score: {fold_f1:.4f}")
        
        # Test predictions for this fold
        fold_preds = model.predict(X_test, num_iteration=model.best_iteration)
        fold_test_preds[fold] = fold_preds
        
        # Print feature importance for each fold
        importance = model.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        print(f"\nTop 10 important features for fold {fold}:")
        print(feature_importance_df.head(10))
    
    # Create result dataframes
    oof_result = oof_stacked.copy()
    test_result = test_stacked.copy()
    
    # Add OOF model probabilities
    for i, col in enumerate(prob_cols):
        oof_result[col] = oof_preds[:, i]
    
    # Update predicted class for OOF
    oof_result['predicted_class'] = np.argmax(oof_preds, axis=1)
    
    # For test predictions, we use the fold-specific predictions
    # Store fold-specific predictions in the test_result dataframe
    for fold in range(n_folds):
        for i, col in enumerate(prob_cols):
            test_result[f"{col}_fold_{fold}"] = fold_test_preds[fold][:, i]
    
    # Add fold-specific predictions for each fold
    for fold in range(n_folds):
        test_result[f"predicted_class_fold_{fold}"] = np.argmax(fold_test_preds[fold], axis=1)
    
    # We also add the average prediction for reference
    avg_test_preds = np.mean([fold_test_preds[fold] for fold in range(n_folds)], axis=0)
    for i, col in enumerate(prob_cols):
        test_result[col] = avg_test_preds[:, i]
    
    test_result['predicted_class'] = np.argmax(avg_test_preds, axis=1)
    
    # Map back to original gas labels
    reverse_gas_mapping = {v: k for k, v in gas_mapping.items()}
    
    # For OOF
    if 'Gas' in oof_result.columns:
        oof_result['predicted_gas'] = oof_result['predicted_class'].map(reverse_gas_mapping)
    elif 'predicted_label' in oof_result.columns:
        gas_types = [col.replace('prob_', '') for col in prob_cols]
        oof_result['predicted_label'] = [gas_types[i] for i in oof_result['predicted_class']]
    
    # For Test - add for each fold
    # Determine gas class names from prob_cols
    gas_types = [col.replace('prob_', '') for col in prob_cols]
    
    # Map fold-specific predictions to gas classes
    for fold in range(n_folds):
        fold_pred_col = f"predicted_class_fold_{fold}"
        test_result[f"predicted_gas_fold_{fold}"] = test_result[fold_pred_col].apply(lambda x: gas_types[x])
    
    # Map average prediction to gas class
    test_result['predicted_gas'] = test_result['predicted_class'].apply(lambda x: gas_types[x])
    
    return oof_result, test_result

# Main execution
if all_oof_dfs and all_test_dfs:
    # Get probability column names from the first OOF df
    prob_cols = [col for col in all_oof_dfs[0][1].columns if col.startswith('prob_')]
    
    # Perform stacking with fold-specific test predictions
    print("\n=== Improved Stacking with LightGBM (exp28) ===")
    print("Performing stacking with fold-specific test predictions...")
    oof_stacked_df, test_stacked_df = stacking_lightgbm(all_oof_dfs, all_test_dfs, prob_cols)
    
    if oof_stacked_df is not None:
        # Evaluate OOF performance
        y_true = oof_stacked_df['Gas']
        y_pred = oof_stacked_df['predicted_gas']
        
        # Get unique gas classes
        gas_classes = sorted(y_true.unique())
        
        # For metrics calculation, we need numeric labels
        le = LabelEncoder()
        le.fit(gas_classes)
        y_true_encoded = le.transform(y_true)
        y_pred_encoded = le.transform(y_pred)
        
        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
        
        print(f"LightGBM Stacking OOF Accuracy: {accuracy:.4f}")
        print(f"LightGBM Stacking OOF F1 Score: {f1:.4f}")
        
        print("\nClassification Report (LightGBM Stacking):")
        print(classification_report(y_true_encoded, y_pred_encoded, target_names=gas_classes))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=gas_classes, yticklabels=gas_classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Improved LightGBM Stacking)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'improved_stacking_confusion_matrix.png'))
        plt.close()
    
    # Save improved stacking ensemble predictions
    if oof_stacked_df is not None:
        oof_stacking_path = os.path.join(OUTPUT_DIR, 'oof_stacking_predictions.csv')
        oof_stacked_df.to_csv(oof_stacking_path, index=False)
        print(f"Saved LightGBM stacking OOF predictions to {oof_stacking_path}")
    
    if test_stacked_df is not None:
        test_stacking_path = os.path.join(OUTPUT_DIR, 'test_stacking_predictions.csv')
        test_stacked_df.to_csv(test_stacking_path, index=False)
        print(f"Saved LightGBM stacking test predictions with fold-specific results to {test_stacking_path}")
        
        # Create submission files from fold-specific predictions
        n_folds = len([col for col in test_stacked_df.columns if col.startswith('predicted_class_fold_')])
        print(f"Creating fold-specific submission files for {n_folds} folds")
        
        # Create a submission file for each fold's prediction
        for fold in range(n_folds):
            fold_gas_col = f"predicted_gas_fold_{fold}"
            
            if fold_gas_col in test_stacked_df.columns:
                # Create submission with index column and prediction
                submission = pd.DataFrame()
                # Use sequential numbering for index
                submission['index'] = range(len(test_stacked_df))
                submission['Gas'] = test_stacked_df[fold_gas_col]
                
                submission_path = os.path.join(OUTPUT_DIR, f'submission_fold_{fold}.csv')
                submission.to_csv(submission_path, index=False)
                print(f"Saved fold {fold} submission file to {submission_path}")
            else:
                print(f"Warning: Could not create submission for fold {fold}, column {fold_gas_col} not found")
        
        # Create a submission file from the average prediction
        if 'predicted_gas' in test_stacked_df.columns:
            submission_avg = pd.DataFrame()
            # Use sequential numbering for index
            submission_avg['index'] = range(len(test_stacked_df))
            submission_avg['Gas'] = test_stacked_df['predicted_gas']
            
            submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
            submission_avg.to_csv(submission_path, index=False)
            print(f"Saved average submission file to {submission_path}")
        else:
            print("Warning: Could not create average submission file")

else:
    print("Not enough data available for improved stacking ensemble.")

print("\nImproved ensemble experiment (exp28) complete!")
