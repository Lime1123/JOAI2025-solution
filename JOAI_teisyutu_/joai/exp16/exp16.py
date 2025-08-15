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
import timm
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
OUTPUT_DIR = 'joai/exp16'  # Directory to save models and outputs

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)

# Define constants
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_FOLDS = 5  # Number of folds for K-Fold CV
PRELOAD_IMAGES = True  # Flag to control image preloading

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

# Create a dataset class with features converted to images
class GasDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, is_test=False, preload=True, feature_stats=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.preload = preload
        self.feature_stats = feature_stats  # Statistics for standardization
        self.images = {}
        
        if self.preload:
            print(f"Preloading {len(self.df)} images...")
            for idx in range(len(self.df)):
                img_name = self.df.iloc[idx]['image_path_uuid']
                img_path = os.path.join(self.image_dir, img_name)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = np.array(image)
                    self.images[img_name] = image
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    # Store a blank image as a fallback
                    self.images[img_name] = np.zeros((224, 224, 3), dtype=np.uint8)
            print("Image preloading complete.")
        
    def __len__(self):
        return len(self.df)
    
    def _add_feature_channels(self, img, idx):
        """Add temp_min, temp_max, temp_unit, and has_coordinates as additional channels"""
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
        
        # Ensure image is 224x224x3
        if img.shape[:2] != (224, 224):
            img = np.array(Image.fromarray(img).resize((224, 224)))
        
        # Standardize temperature values using precomputed statistics
        temp_min_standardized = (float(temp_min) - self.feature_stats['temp_min_mean']) / self.feature_stats['temp_min_std']
        temp_max_standardized = (float(temp_max) - self.feature_stats['temp_max_mean']) / self.feature_stats['temp_max_std']
        
        # Create feature channels with the same shape as the image
        temp_min_channel = np.ones(img.shape[:2], dtype=np.float32) * temp_min_standardized
        temp_max_channel = np.ones(img.shape[:2], dtype=np.float32) * temp_max_standardized
        temp_unit_channel = np.ones(img.shape[:2], dtype=np.float32) * (temp_unit_val / 2)  # Normalize to [0, 1]
        has_coordinates_channel = np.ones(img.shape[:2], dtype=np.float32) * has_coordinates_val
        
        # Return image with additional channels
        return img, temp_min_channel, temp_max_channel, temp_unit_channel, has_coordinates_channel
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_path_uuid']
        
        # Get image from preloaded dict or load it on-the-fly
        if self.preload and img_name in self.images:
            image = self.images[img_name]
        else:
            img_path = os.path.join(self.image_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a blank image as a fallback
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Get image with feature channels
        img, temp_min_channel, temp_max_channel, temp_unit_channel, has_coordinates_channel = self._add_feature_channels(image, idx)
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        
        # In PyTorch, images are [C, H, W]
        # Convert channels to tensor format
        temp_min_tensor = torch.tensor(temp_min_channel, dtype=torch.float32).unsqueeze(0)
        temp_max_tensor = torch.tensor(temp_max_channel, dtype=torch.float32).unsqueeze(0)
        temp_unit_tensor = torch.tensor(temp_unit_channel, dtype=torch.float32).unsqueeze(0)
        has_coordinates_tensor = torch.tensor(has_coordinates_channel, dtype=torch.float32).unsqueeze(0)
        
        # Concatenate all channels
        # img is already a tensor with shape [3, H, W] after transform
        multi_channel_img = torch.cat([img, temp_min_tensor, temp_max_tensor, temp_unit_tensor, has_coordinates_tensor], dim=0)
        
        if self.is_test:
            return multi_channel_img
        else:
            label = label_map[self.df.iloc[idx]['Gas']]
            return multi_channel_img, label

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
        self.train_dataset = GasDataset(self.train_data, self.image_dir, transform=train_transform, 
                                       preload=self.preload, feature_stats=self.feature_stats)
        self.val_dataset = GasDataset(self.val_data, self.image_dir, transform=test_transform, 
                                     preload=self.preload, feature_stats=self.feature_stats)
        if self.test_df is not None:
            self.test_dataset = GasDataset(self.test_df, self.image_dir, transform=test_transform, 
                                          is_test=True, preload=self.preload, feature_stats=self.feature_stats)
    
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

# Define transformations with Albumentations - only for RGB part
train_transform = A.Compose([
    A.Resize(224, 224),
    #A.HorizontalFlip(p=0.5),
    #A.RandomRotate90(p=1),
    #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.75),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Define the Lightning Module for the CNN model with multi-channel input
class MultiChannelGasClassifierModule(pl.LightningModule):
    def __init__(self, num_classes=4, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained ResNet50 model from timm, modify first convolution layer for multi-channel input
        self.backbone = timm.create_model('resnet50.a1h_in1k', pretrained=True)
        
        # Replace the first conv layer to accept 7 channels (3 RGB + 4 feature channels)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            7, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize the new layer with weights from the pre-trained model, extended for additional channels
        with torch.no_grad():
            # Copy the weights for the RGB channels
            self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight.clone()
            
            # Initialize the weights for the additional channels as the average of RGB weights
            rgb_mean = original_conv.weight.mean(dim=1, keepdim=True)
            self.backbone.conv1.weight[:, 3:, :, :] = rgb_mean.expand(-1, 4, -1, -1)
        
        # Replace final classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.),
            nn.Linear(512, num_classes)
        )
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Track metrics
        self.train_preds,  self.train_targets = [], []
        self.val_preds,    self.val_targets   = [], []

        self.train_acc = []
        self.val_acc = []
        self.train_f1 = []
        self.val_f1 = []
    
    def forward(self, x):
        return self.backbone(x)
    
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
        x, y = batch
        logits = self(x)
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
        x, y = batch
        logits = self(x)
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
        # For test set, batch is just images without labels
        x = batch
        
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return preds, probs
    
    def on_train_epoch_end(self):
        # Calculate F1 score across the epoch
        all_preds   = torch.cat(self.train_preds).cpu().numpy()
        all_targets = torch.cat(self.train_targets).cpu().numpy()
        f1 = f1_score(all_targets, all_preds, average='weighted')
        self.train_f1.append(f1)
        self.log('train_f1', f1, prog_bar=True)
        self.train_preds.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self):
        # Calculate F1 score across the epoch
        all_preds   = torch.cat(self.val_preds).cpu().numpy()
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

# Function to get predictions with logits
def get_predictions(model, dataloader, device):
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list) and len(batch) == 2 and not isinstance(batch[0], list):
                # Handle validation set where batch = [x, y]
                x, _ = batch
            else:
                # Handle test set where batch is just x
                x = batch
            
            # Move input to device
            if isinstance(x, list):
                x = [i.to(device) for i in x]
            else:
                x = x.to(device)
            
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
    
    return np.vstack(all_logits)

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

# Collect metrics across folds
cv_scores = {'accuracy': [], 'f1': []}

# Train models for each fold
for fold in range(NUM_FOLDS):
    print(f"\n{'='*50}\nTraining Fold {fold+1}/{NUM_FOLDS}\n{'='*50}")
    
    # Create data module for this fold
    data_module = GasDataModule(train_df, test_df, IMAGE_DIR, fold_idx=fold, 
                              batch_size=BATCH_SIZE, preload=PRELOAD_IMAGES, feature_stats=feature_stats)
    data_module.setup()
    
    # Get fold data sizes
    train_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index
    print(f"  Train: {len(train_idx)} samples, Validation: {len(val_idx)} samples")
    
    # Create model
    model = MultiChannelGasClassifierModule(num_classes=4, learning_rate=LEARNING_RATE)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_DIR, 'checkpoints', f'fold_{fold}'),
        filename='gas-classifier-{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_f1',
        mode='max'
    )
    
    #early_stop_callback = EarlyStopping(
    #    monitor='val_f1',
    #    patience=5,
    #    verbose=True,
    #    mode='max'
    #)
    
    # Create logger
    logger = TensorBoardLogger(os.path.join(OUTPUT_DIR, 'lightning_logs'), name=f'gas_classifier_fold_{fold}')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback],# early_stop_callback],
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
    #model_save_path = os.path.join(OUTPUT_DIR, 'models', f'fold_{fold}_best_model.pth')
    #torch.save(model.state_dict(), model_save_path)
    #print(f"Saved model weights to {model_save_path}")
    
    # Load best model for predictions
    best_model = MultiChannelGasClassifierModule.load_from_checkpoint(best_model_path)
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_map.keys()), 
                yticklabels=list(label_map.keys()))
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
