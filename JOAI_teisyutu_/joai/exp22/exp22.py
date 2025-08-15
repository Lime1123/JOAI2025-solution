import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Set random seeds for reproducibility
pl.seed_everything(42)

# Define paths
DATA_DIR = 'data'  # Update this path if your data is located elsewhere
OUTPUT_DIR = 'joai/exp22'  # Directory to save models and outputs

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)

# Define constants
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_FOLDS = 5  # Number of folds for K-Fold CV
PRELOAD_IMAGES = True  # Flag to control image preloading
IMAGE_SIZE = 224

# Print configuration
print(f"Configuration:")
print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  CV Folds: {NUM_FOLDS}")
print(f"  Preload Images: {PRELOAD_IMAGES}")

# Load data
print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_folds_temp.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_temp.csv'))

IMAGE_DIR = os.path.join(DATA_DIR, 'images')
print(f"Training data shape: {train_df.shape}")

# Compute statistics for feature normalization from the entire training dataset
print("Computing feature statistics for normalization...")
# Handle possible NaN values in numeric columns
temp_min_mean = train_df['temp_min'].replace([np.inf, -np.inf], np.nan).mean(skipna=True)
temp_max_mean = train_df['temp_max'].replace([np.inf, -np.inf], np.nan).mean(skipna=True)

# Fill NaN values with means
train_df['temp_min'] = train_df['temp_min'].replace([np.inf, -np.inf], np.nan).fillna(temp_min_mean)
train_df['temp_max'] = train_df['temp_max'].replace([np.inf, -np.inf], np.nan).fillna(temp_max_mean)

if test_df is not None:
    test_df['temp_min'] = test_df['temp_min'].replace([np.inf, -np.inf], np.nan).fillna(temp_min_mean)
    test_df['temp_max'] = test_df['temp_max'].replace([np.inf, -np.inf], np.nan).fillna(temp_max_mean)

# Calculate statistics for standardization
temp_min_std = train_df['temp_min'].std()
temp_max_std = train_df['temp_max'].std()

# Avoid division by zero
if temp_min_std == 0:
    temp_min_std = 1.0
if temp_max_std == 0:
    temp_max_std = 1.0

feature_stats = {
    'temp_min_mean': temp_min_mean,
    'temp_min_std': temp_min_std,
    'temp_max_mean': temp_max_mean,
    'temp_max_std': temp_max_std
}

print(f"Feature statistics: {feature_stats}")

# Check the distribution of target classes
print("Target distribution:")
print(train_df['Gas'].value_counts())

# Create label mapping
label_map = {
    'NoGas': 0,
    'Perfume': 1,
    'Smoke': 2,
    'Mixture': 3
}
inv_label_map = {v: k for k, v in label_map.items()}
print("Label mapping:", label_map)

# Initialize CLIP processor for data preprocessing
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create a dataset class that includes both image and metadata
class GasDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, is_test=False, preload=True, feature_stats=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.preload = preload
        self.feature_stats = feature_stats
        self.images = {}
        self.processor = clip_processor
        
        if self.preload:
            print(f"Preloading {len(self.df)} images...")
            for idx in range(len(self.df)):
                img_name = self.df.iloc[idx]['image_path_uuid']
                img_path = os.path.join(self.image_dir, img_name)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.images[img_name] = image
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    # Store a blank image as a fallback
                    self.images[img_name] = Image.new('RGB', (224, 224), color='black')
            print("Image preloading complete.")
        
    def __len__(self):
        return len(self.df)
    
    def _get_normalized_features(self, idx):
        """Extract and normalize metadata features"""
        row = self.df.iloc[idx]
        
        # Get feature values
        temp_min = row.get('temp_min', self.feature_stats['temp_min_mean'])
        temp_max = row.get('temp_max', self.feature_stats['temp_max_mean'])
        temp_unit = row.get('temp_unit', '')
        has_coordinates = row.get('has_coordinates', False)
        
        # Handle NaN values
        if pd.isna(temp_min) or np.isinf(temp_min):
            temp_min = self.feature_stats['temp_min_mean']
        if pd.isna(temp_max) or np.isinf(temp_max):
            temp_max = self.feature_stats['temp_max_mean']
        
        # Convert temp_unit to numeric value
        if temp_unit == 'C':
            temp_unit_val = 0
        elif temp_unit == 'F':
            temp_unit_val = 1
        else:
            temp_unit_val = 2  # For any other value
        
        # Convert has_coordinates to numeric value
        has_coordinates_val = 1 if has_coordinates else 0
        
        # Standardize temperature values using precomputed statistics
        temp_min_standardized = (float(temp_min) - self.feature_stats['temp_min_mean']) / self.feature_stats['temp_min_std']
        temp_max_standardized = (float(temp_max) - self.feature_stats['temp_max_mean']) / self.feature_stats['temp_max_std']
        
        # Normalize temp_unit value to [0, 1]
        temp_unit_normalized = temp_unit_val / 2
        
        # Return all features as a tensor
        features = torch.tensor([
            temp_min_standardized,
            temp_max_standardized,
            temp_unit_normalized,
            has_coordinates_val
        ], dtype=torch.float32)
        
        return features
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_path_uuid']
        
        # Get image from preloaded dict or load it on-the-fly
        if self.preload and img_name in self.images:
            image = self.images[img_name]
        else:
            img_path = os.path.join(self.image_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a blank image as a fallback
                image = Image.new('RGB', (224, 224), color='black')
        
        # Process image using CLIP processor
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].squeeze()
        
        # Get metadata features
        features = self._get_normalized_features(idx)
        
        if self.is_test:
            return pixel_values, features
        else:
            label = label_map[self.df.iloc[idx]['Gas']]
            return pixel_values, features, label

def softmax(x, axis=1):
    # Subtracting the maximum for numerical stability
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Lightning Data Module for K-Fold CV
class GasDataModule(pl.LightningDataModule):
    def __init__(self, df, test_df, image_dir, fold_idx=None, batch_size=32, preload=True, feature_stats=None):
        super().__init__()
        self.df = df
        self.test_df = test_df
        self.image_dir = image_dir
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.preload = preload
        self.feature_stats = feature_stats
    
    def setup(self, stage=None):
        if self.fold_idx is not None:
            # For K-Fold CV
            train_idx = self.df[self.df['fold'] != self.fold_idx].index
            val_idx = self.df[self.df['fold'] == self.fold_idx].index
            
            self.train_data = self.df.loc[train_idx].reset_index(drop=True)
            self.val_data = self.df.loc[val_idx].reset_index(drop=True)
        else:
            # For final training on all data
            self.train_data, self.val_data = train_test_split(
                self.df, test_size=0.2, random_state=42, stratify=self.df['Gas']
            )
        
        # Create datasets
        self.train_dataset = GasDataset(self.train_data, self.image_dir, preload=self.preload, feature_stats=self.feature_stats)
        self.val_dataset = GasDataset(self.val_data, self.image_dir, preload=self.preload, feature_stats=self.feature_stats)
        if self.test_df is not None:
            self.test_dataset = GasDataset(self.test_df, self.image_dir, is_test=True, preload=self.preload, feature_stats=self.feature_stats)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        if self.test_df is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
        return None
    
    def get_val_dataset(self):
        return self.val_data, self.val_dataset

# Define the Lightning Module for the ViT model
class CLIPVitWithMetadataModule(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained CLIP ViT model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze the CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get the dimension of the visual projection
        self.feature_dim = self.clip_model.visual_projection.out_features
        
        # Metadata feature dimension (4 features)
        self.metadata_dim = 4
        
        # Fusion of ViT features and metadata features
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim + self.metadata_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.),
            nn.Linear(512, num_classes)
        )
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Track metrics
        self.train_preds, self.train_targets = [], []
        self.val_preds, self.val_targets = [], []
        
        self.train_acc = []
        self.val_acc = []
        self.train_f1 = []
        self.val_f1 = []
    
    def forward(self, x, metadata):
        # Extract features from the visual encoder
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(
                pixel_values=x
            )
            image_embeds = vision_outputs[1]
            image_features = self.clip_model.visual_projection(image_embeds)
        
        # Concatenate image features with metadata
        combined_features = torch.cat([image_features, metadata], dim=1)
        
        # Apply our fusion classifier
        return self.fusion(combined_features)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS+1, eta_min=0.0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        x, metadata, y = batch
        logits = self(x, metadata)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.train_preds.append(preds)
        self.train_targets.append(y)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'preds': preds, 'targets': y}
    
    def validation_step(self, batch, batch_idx):
        x, metadata, y = batch
        logits = self(x, metadata)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.val_preds.append(preds)
        self.val_targets.append(y)

        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        # Return probabilities for OOF predictions
        return {'val_loss': loss, 'preds': preds, 'targets': y, 'probs': F.softmax(logits, dim=1)}
    
    def predict_step(self, batch, batch_idx):
        # For test set, batch is images and metadata without labels
        x, metadata = batch
        
        logits = self(x, metadata)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return preds, probs
    
    def on_train_epoch_end(self):
        # Calculate F1 score across the epoch
        all_preds = torch.cat(self.train_preds).cpu().numpy()
        all_targets = torch.cat(self.train_targets).cpu().numpy()
        f1 = f1_score(all_targets, all_preds, average='weighted')
        self.train_f1.append(f1)
        self.log('train_f1', f1, prog_bar=True)
        self.train_preds.clear()
        self.train_targets.clear()
    
    def on_validation_epoch_end(self):
        # Calculate F1 score across the epoch
        all_preds = torch.cat(self.val_preds).cpu().numpy()
        all_targets = torch.cat(self.val_targets).cpu().numpy()
        
        f1 = f1_score(all_targets, all_preds, average='weighted')
        self.val_f1.append(f1)
        self.log('val_f1', f1, prog_bar=True)
        self.val_preds.clear()
        self.val_targets.clear()
        
        return {'val_f1': f1, 'val_preds': all_preds, 'val_targets': all_targets}
    
    def get_metrics(self):
        return {
            'train_f1': self.train_f1,
            'val_f1': self.val_f1
        }

def get_predictions(model, dataloader, device):
    model.to(device)
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # For validation data (image, metadata, label)
                x, metadata, _ = batch
            else:  # For test data (image, metadata)
                x, metadata = batch
            
            x = x.to(device)
            metadata = metadata.to(device)
            
            logits = model(x, metadata)
            all_logits.append(logits.cpu().numpy())
    
    return np.vstack(all_logits)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize OOF predictions array (for logits)
oof_logits = np.zeros((len(train_df), len(label_map)))

# Initialize array for test predictions (for logits)
if test_df is not None:
    test_logits = np.zeros((len(test_df), len(label_map)))

# Print training information
print("\nPerforming K-Fold Cross-validation...")
print(f"Number of folds: {NUM_FOLDS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Preload images: {PRELOAD_IMAGES}")
print(f"Model: CLIP ViT with metadata fusion")

# Collect metrics across folds
cv_scores = {'accuracy': [], 'f1': []}

# Train models for each fold
for fold in range(NUM_FOLDS):
    print(f"\n{'='*50}\nTraining Fold {fold+1}/{NUM_FOLDS}\n{'='*50}")
    
    # Create data module for this fold
    data_module = GasDataModule(
        train_df, test_df, IMAGE_DIR, fold_idx=fold, batch_size=BATCH_SIZE, 
        preload=PRELOAD_IMAGES, feature_stats=feature_stats
    )
    data_module.setup()
    
    # Get fold data sizes
    train_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index
    print(f"  Train: {len(train_idx)} samples, Validation: {len(val_idx)} samples")
    
    # Create model
    model = CLIPVitWithMetadataModule(num_classes=4, learning_rate=LEARNING_RATE)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_DIR, 'checkpoints', f'fold_{fold}'),
        filename='clip-vit-metadata-{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_f1',
        mode='max'
    )
    
    # Create logger
    logger = TensorBoardLogger(os.path.join(OUTPUT_DIR, 'lightning_logs'), name=f'clip_vit_metadata_fold_{fold}')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        log_every_n_steps=1,
        gradient_clip_val=1.0,
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Get best model path
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path for fold {fold}: {best_model_path}")
    
    # Save best model weights
    model_save_path = os.path.join(OUTPUT_DIR, 'models', f'fold_{fold}_best_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model weights to {model_save_path}")
    
    # Load best model for predictions
    best_model = CLIPVitWithMetadataModule.load_from_checkpoint(best_model_path)
    best_model.eval()
    
    # Get validation data for this fold
    val_data, val_dataset = data_module.get_val_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Make OOF predictions for this fold (with logits)
    val_logits_fold = get_predictions(best_model, val_dataloader, best_model.device)
    oof_logits[val_idx] = val_logits_fold
    
    # Convert logits to probabilities for metrics calculation
    val_probs = softmax(val_logits_fold, axis=1)
    val_pred_class = np.argmax(val_probs, axis=1)
    y_val = val_data['Gas'].map(label_map).values
    
    accuracy = accuracy_score(y_val, val_pred_class)
    f1 = f1_score(y_val, val_pred_class, average='weighted')
    
    cv_scores['accuracy'].append(accuracy)
    cv_scores['f1'].append(f1)
    
    print(f"  Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Plot confusion matrix for this fold
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_val, val_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_fold_{fold}.png'))
    plt.close()
    
    # Make test predictions for this fold if test data exists (with logits)
    if test_df is not None:
        test_dataloader = data_module.test_dataloader()
        test_logits_fold = get_predictions(best_model, test_dataloader, best_model.device)
        test_logits += test_logits_fold / NUM_FOLDS

# Convert averaged logits to probabilities
oof_preds = softmax(oof_logits, axis=1)

# Create OOF dataframe with probabilities
oof_df = train_df.copy()
for i, label in enumerate(label_map.keys()):
    oof_df[f'prob_{label}'] = oof_preds[:, i]

# Add predicted class
oof_df['predicted_class_idx'] = np.argmax(oof_preds, axis=1)
oof_df['predicted_class'] = oof_df['predicted_class_idx'].map(inv_label_map)

# Calculate OOF metrics
oof_pred_class = np.argmax(oof_preds, axis=1)
y_true = oof_df['Gas'].map(label_map).values

oof_accuracy = accuracy_score(y_true, oof_pred_class)
oof_f1 = f1_score(y_true, oof_pred_class, average='weighted')

print("\nOut-of-fold prediction results:")
print(f"OOF Accuracy: {oof_accuracy:.4f}")
print(f"OOF F1 Score: {oof_f1:.4f}")

# Save OOF predictions
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_predictions_probs.csv'), index=False)
print(f"OOF predictions with probabilities saved to {os.path.join(OUTPUT_DIR, 'oof_predictions_probs.csv')}")

print("\nCross-validation results:")
print(f"Mean Accuracy: {np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}")
print(f"Mean F1 Score: {np.mean(cv_scores['f1']):.4f} ± {np.std(cv_scores['f1']):.4f}")

print("\nClassification Report (OOF predictions):")
print(classification_report(y_true, oof_pred_class, target_names=list(label_map.keys())))

# Plot overall confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, oof_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(label_map.keys()), 
            yticklabels=list(label_map.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Overall Confusion Matrix (OOF Predictions)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overall_confusion_matrix.png'))
plt.close()

# Plot OOF distribution
plt.figure(figsize=(12, 8))
for i, label in enumerate(label_map.keys()):
    plt.subplot(2, 2, i+1)
    plt.hist(oof_preds[:, i], bins=50)
    plt.title(f'OOF Probability Distribution - {label}')
    plt.xlabel('Probability')
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'oof_distribution.png'))
plt.close()

# Create test submission with final ensemble predictions if test data exists
if test_df is not None:
    # Convert averaged test logits to probabilities
    test_preds = softmax(test_logits, axis=1)
    
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_pred_labels = [inv_label_map[pred] for pred in test_pred_classes]

    # Create submission file
    submission = pd.DataFrame({
        'index': range(len(test_pred_labels)),
        'Gas': test_pred_labels
    })

    # Add probability columns to submission
    for i, label in enumerate(label_map.keys()):
        submission[f'prob_{label}'] = test_preds[:, i]

    submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"Submission file saved to {os.path.join(OUTPUT_DIR, 'submission.csv')}")
    
    # Create and save test_predictions_probs.csv with more details
    test_predictions_df = test_df.copy()
    # Add probability columns
    for i, label in enumerate(label_map.keys()):
        test_predictions_df[f'prob_{label}'] = test_preds[:, i]
    
    # Add predicted class columns
    test_predictions_df['predicted_class_idx'] = test_pred_classes
    test_predictions_df['predicted_class'] = test_pred_labels
    
    # Save test predictions with probabilities
    test_predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv'), index=False)
    print(f"Test predictions with probabilities saved to {os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv')}")

print("Training with K-Fold cross-validation completed successfully!")
