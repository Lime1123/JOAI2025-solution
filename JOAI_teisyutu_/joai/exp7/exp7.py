import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define paths
DATA_DIR = 'data'
OUTPUT_DIR = 'joai/exp7'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom dataset class
class GasDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        hidden_size = 64*4*2*2*2
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),

            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, meta_feature):
        output = self.fc(meta_feature)
        return output

# Function to train a model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=20):
    model.to(device)
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

# Function to evaluate the model
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1, np.array(all_preds), np.array(all_labels), np.array(all_probs), np.array(all_logits)

# Main execution
if __name__ == "__main__":
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
    #plt.savefig(os.path.join(OUTPUT_DIR, 'temp_min_max_scatter.png'))
    plt.close()
    
    # Explore the relationship between MQ8, MQ5 and Gas
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='MQ8', y='MQ5', hue='Gas', data=train_df)
    plt.title('MQ8 vs MQ5 by Gas Type')
    #plt.savefig(os.path.join(OUTPUT_DIR, 'mq8_mq5_scatter.png'))
    plt.close()
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    train_df['Gas_encoded'] = label_encoder.fit_transform(train_df['Gas'])
    
    # Map encoded values back to original labels for reference
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print("Label mapping:", label_mapping)
    
    # Define the features and target
    X = train_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit_encoded', 'has_coordinates_encoded']].values
    y = train_df['Gas_encoded']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    input_size = 6  # MQ8, MQ5, temp_min, temp_max, temp_unit_encoded, has_coordinates_encoded
    hidden_size = 12  # Increase hidden size for more features
    num_classes = len(label_encoder.classes_)
    batch_size = 16
    learning_rate = 0.01
    
    # K-Fold Cross-validation using pre-defined folds
    print("\nPerforming K-Fold Cross-validation...")
    n_folds = train_df['fold'].nunique()
    cv_scores = {'accuracy': [], 'f1': []}
    oof_predictions = np.zeros((len(train_df), num_classes))
    oof_logits = np.zeros((len(train_df), num_classes))
    fold_models = []
    
    for fold in range(n_folds):
        print(f"Training fold {fold + 1}/{n_folds}")
        
        # Split data using the fold column
        train_idx = train_df[train_df['fold'] != fold].index
        val_idx = train_df[train_df['fold'] == fold].index
        
        X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        y_train_fold, y_val_fold = y.loc[train_idx].values, y.loc[val_idx].values
        
        print(f"  Train: {len(X_train_fold)} samples, Validation: {len(X_val_fold)} samples")
        
        # Create datasets and dataloaders
        train_dataset = GasDataset(X_train_fold, y_train_fold)
        val_dataset = GasDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize the model
        model = SimpleNN(input_size, hidden_size, num_classes)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, device
        )
        
        # Store the model
        fold_models.append(trained_model)
        
        # Save model
        model_path = os.path.join(OUTPUT_DIR, f'nn_model_fold_{fold}.pt')
        torch.save(trained_model.state_dict(), model_path)
        print(f"  Model saved to {model_path}")
        
        # Evaluate the model
        accuracy, f1, val_pred_class, val_true, val_probs, val_logits = evaluate_model(
            trained_model, val_loader, device
        )
        
        # Store predictions for this fold
        oof_predictions[val_idx] = val_probs
        oof_logits[val_idx] = val_logits
        
        cv_scores['accuracy'].append(accuracy)
        cv_scores['f1'].append(f1)
        
        print(f"  Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        # Plot confusion matrix for this fold
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(val_true, val_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Fold {fold + 1}')
        plt.tight_layout()
        #plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_fold_{fold}.png'))
        plt.close()
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - Fold {fold + 1}')
        plt.legend()
        #plt.savefig(os.path.join(OUTPUT_DIR, f'loss_fold_{fold}.png'))
        plt.close()
        
        # Plot fold-specific scatter plots for important feature pairs
        plt.figure(figsize=(10, 8))
        val_scatter_x = X_scaled[val_idx][:, 0]  # MQ8 (scaled)
        val_scatter_y = X_scaled[val_idx][:, 1]  # MQ5 (scaled)
        val_y_true = y_val_fold
        
        # True labels
        for i, gas_type in enumerate(label_encoder.classes_):
            mask = (val_y_true == i)
            if np.any(mask):
                plt.scatter(val_scatter_x[mask], val_scatter_y[mask], 
                           c=f'C{i}', marker='o', label=f'{gas_type} (True)', alpha=0.7)
        
        plt.title(f'MQ8 vs MQ5 by Gas Type (True Values) - Fold {fold + 1}')
        plt.xlabel('MQ8 (scaled)')
        plt.ylabel('MQ5 (scaled)')
        plt.legend()
        #plt.savefig(os.path.join(OUTPUT_DIR, f'mq_scatter_true_fold_{fold}.png'))
        plt.close()
        
        # Predicted labels
        plt.figure(figsize=(10, 8))
        for i, gas_type in enumerate(label_encoder.classes_):
            mask = (val_pred_class == i)
            if np.any(mask):
                plt.scatter(val_scatter_x[mask], val_scatter_y[mask], 
                           c=f'C{i}', marker='x', label=f'{gas_type} (Predicted)', alpha=0.7)
        
        plt.title(f'MQ8 vs MQ5 by Gas Type (Predictions) - Fold {fold + 1}')
        plt.xlabel('MQ8 (scaled)')
        plt.ylabel('MQ5 (scaled)')
        plt.legend()
        #plt.savefig(os.path.join(OUTPUT_DIR, f'mq_scatter_pred_fold_{fold}.png'))
        plt.close()
    
    # Calculate OOF metrics
    oof_pred_class = np.argmax(oof_predictions, axis=1)
    oof_accuracy = accuracy_score(y, oof_pred_class)
    oof_f1 = f1_score(y, oof_pred_class, average='weighted')
    
    # Save OOF probabilities to CSV
    oof_probs_df = pd.DataFrame(train_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit', 'has_coordinates', 'Gas', 'Gas_encoded', 'fold']].copy())
    for i, gas in enumerate(label_encoder.classes_):
        oof_probs_df[f'prob_{gas}'] = oof_predictions[:, i]
    oof_probs_df['predicted_class'] = oof_pred_class
    oof_probs_df['predicted_gas'] = label_encoder.inverse_transform(oof_pred_class)
    oof_probs_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_predictions_probs.csv'), index=False)
    print(f"OOF probabilities saved to {os.path.join(OUTPUT_DIR, 'oof_predictions_probs.csv')}")
    
    print("\nOut-of-fold prediction results:")
    print(f"OOF Accuracy: {oof_accuracy:.4f}")
    print(f"OOF F1 Score: {oof_f1:.4f}")
    
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
    
    # Visualize decision boundaries using the average prediction from all folds
    plt.figure(figsize=(12, 10))
    
    # Select the two features for visualization (MQ8 and MQ5)
    feature_names = ['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit_encoded', 'has_coordinates_encoded']
    visual_features = [0, 1]  # MQ8, MQ5 indices in X_scaled
    
    # Create a mesh grid for plotting decision boundaries
    x_min, x_max = X_scaled[:, visual_features[0]].min() - 1, X_scaled[:, visual_features[0]].max() + 1
    y_min, y_max = X_scaled[:, visual_features[1]].min() - 1, X_scaled[:, visual_features[1]].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Get predictions for all points in the mesh from all models
    mesh_preds = np.zeros((xx.ravel().shape[0], num_classes))
    
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            # Create full feature vectors using median values for non-visualization features
            mesh_inputs = np.zeros((xx.ravel().shape[0], X_scaled.shape[1]))
            
            # Fill with median values
            for i in range(X_scaled.shape[1]):
                if i not in visual_features:
                    mesh_inputs[:, i] = np.median(X_scaled[:, i])
            
            # Put visualization features in the mesh grid
            mesh_inputs[:, visual_features[0]] = xx.ravel()
            mesh_inputs[:, visual_features[1]] = yy.ravel()
            
            mesh_tensor = torch.tensor(mesh_inputs, dtype=torch.float32).to(device)
            outputs = model(mesh_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            mesh_preds += probs
            
    mesh_preds /= len(fold_models)  # Average predictions
    
    Z = np.argmax(mesh_preds, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    # Plot training points
    scatter_x = X_scaled[:, visual_features[0]]
    scatter_y = X_scaled[:, visual_features[1]]
    
    for i, gas_type in enumerate(label_encoder.classes_):
        idx = train_df[train_df['Gas'] == gas_type].index
        plt.scatter(scatter_x[idx], scatter_y[idx], 
                    c=f'C{i}', label=gas_type, edgecolors='k', alpha=0.7)
    
    plt.title('Decision Boundaries with MQ8 and MQ5 (Average of All Folds)')
    plt.xlabel('MQ8 (scaled)')
    plt.ylabel('MQ5 (scaled)')
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
        
        X_test = test_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit_encoded', 'has_coordinates_encoded']].values
        X_test_scaled = scaler.transform(X_test)
        
        # Create test dataset and dataloader
        test_features = torch.tensor(X_test_scaled, dtype=torch.float32)
        
        # Average predictions from all fold models
        test_probs_ensemble = np.zeros((len(X_test), num_classes))
        for model in fold_models:
            model.eval()
            with torch.no_grad():
                fold_outputs = model(test_features.to(device))
                fold_probs = torch.nn.functional.softmax(fold_outputs, dim=1).cpu().numpy()
                test_probs_ensemble += fold_probs
                
        test_probs_ensemble /= len(fold_models)
        test_pred_class_ensemble = np.argmax(test_probs_ensemble, axis=1)
        
        # Convert numerical predictions back to original labels
        test_pred_labels_ensemble = label_encoder.inverse_transform(test_pred_class_ensemble)
        
        # Save predictions to csv
        submission_df = pd.DataFrame({
            'index': range(len(test_pred_labels_ensemble)),
            'Gas': test_pred_labels_ensemble
        })
        submission_df.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
        print("Submission file saved to:", os.path.join(OUTPUT_DIR, 'submission.csv'))
        
        # Save test prediction probabilities to CSV
        test_probs_df = pd.DataFrame(test_df[['MQ8', 'MQ5', 'temp_min', 'temp_max', 'temp_unit', 'has_coordinates']].copy())
        
        # Add ensemble model probabilities
        for i, gas in enumerate(label_encoder.classes_):
            test_probs_df[f'prob_{gas}'] = test_probs_ensemble[:, i]
        
        # Add predicted labels
        test_probs_df['predicted_label'] = test_pred_labels_ensemble
        
        test_probs_df.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv'), index=False)
        print(f"Test probabilities saved to {os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv')}")

    print("Experiment completed successfully!")
