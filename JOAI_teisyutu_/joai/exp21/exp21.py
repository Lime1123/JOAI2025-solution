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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Set random seeds for reproducibility
pl.seed_everything(42)

# Define paths
DATA_DIR = 'data'  # Update this path if your data is located elsewhere
OUTPUT_DIR = 'joai/exp21'  # Directory to save models and outputs

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)

# Define constants
BATCH_SIZE = 32
NUM_EPOCHS_REGRESSION = 5  # Epochs for regression phase
NUM_EPOCHS_CLASSIFICATION = 20  # Epochs for classification phase
LEARNING_RATE_REGRESSION = 1e-4
LEARNING_RATE_CLASSIFICATION = 5e-5
UNFREEZE_EPOCH_REGRESSION = 2  # エポック数2以降でバックボーンの凍結を解除
UNFREEZE_EPOCH_CLASSIFICATION = 2  # エポック数5以降でバックボーンの凍結を解除
NUM_FOLDS = 5  # Number of folds for K-Fold CV
PRELOAD_IMAGES = True  # Flag to control image preloading

# Load data
print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_folds_temp.csv'))
test_df = None

# Check if test.csv exists
test_path = os.path.join(DATA_DIR, 'test.csv')
if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")

IMAGE_DIR = os.path.join(DATA_DIR, 'images')
print(f"Training data shape: {train_df.shape}")

# MQ8とMQ5の標準化のための統計量を計算
mq8_mean = train_df['MQ8'].mean()
mq8_std = train_df['MQ8'].std()
mq5_mean = train_df['MQ5'].mean()
mq5_std = train_df['MQ5'].std()

print(f"MQ8 statistics - Mean: {mq8_mean:.2f}, Std: {mq8_std:.2f}")
print(f"MQ5 statistics - Mean: {mq5_mean:.2f}, Std: {mq5_std:.2f}")

# Check the distribution of target values
print("MQ8 distribution:")
print(train_df['MQ8'].describe())
print("\nMQ5 distribution:")
print(train_df['MQ5'].describe())
print("\nGas distribution:")
print(train_df['Gas'].value_counts())

# Create label mapping for classification
label_map = {
    'NoGas': 0,
    'Perfume': 1,
    'Smoke': 2,
    'Mixture': 3
}
inv_label_map = {v: k for k, v in label_map.items()}
print("Label mapping:", label_map)

# Create a dataset class with preloaded images
class GasDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, is_test=False, preload=True, task='regression',
                 standardize=False, mq8_mean=None, mq8_std=None, mq5_mean=None, mq5_std=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.preload = preload
        self.task = task  # 'regression' or 'classification'
        self.standardize = standardize
        self.mq8_mean = mq8_mean
        self.mq8_std = mq8_std
        self.mq5_mean = mq5_mean
        self.mq5_std = mq5_std
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
        
        if self.transform:
            if self.is_test:
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                transformed = self.transform(image=image)
                image = transformed["image"]
        
        if self.is_test:
            return image
        else:
            if self.task == 'regression':
                # Get MQ8 and MQ5 values as regression targets
                mq8 = self.df.iloc[idx]['MQ8']
                mq5 = self.df.iloc[idx]['MQ5']
                
                # 標準化が有効な場合は値を標準化
                if self.standardize and self.mq8_mean is not None and self.mq8_std is not None and self.mq5_mean is not None and self.mq5_std is not None:
                    mq8 = (mq8 - self.mq8_mean) / self.mq8_std
                    mq5 = (mq5 - self.mq5_mean) / self.mq5_std
                    
                targets = torch.tensor([mq8, mq5], dtype=torch.float32)
                return image, targets
            else:  # classification
                # Get Gas label as classification target
                label = label_map[self.df.iloc[idx]['Gas']]
                return image, label

# Define transformations with Albumentations
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

# Lightning Data Module for K-Fold CV
class GasDataModule(pl.LightningDataModule):
    def __init__(self, df, test_df, image_dir, fold_idx=None, batch_size=32, preload=True, task='regression',
                 standardize=False, mq8_mean=None, mq8_std=None, mq5_mean=None, mq5_std=None):
        super().__init__()
        self.df = df
        self.test_df = test_df
        self.image_dir = image_dir
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.preload = preload
        self.task = task  # 'regression' or 'classification'
        self.standardize = standardize
        self.mq8_mean = mq8_mean
        self.mq8_std = mq8_std
        self.mq5_mean = mq5_mean
        self.mq5_std = mq5_std
    
    def setup(self, stage=None):
        if self.fold_idx is not None:
            # For K-Fold CV
            train_idx = self.df[self.df['fold'] != self.fold_idx].index
            val_idx = self.df[self.df['fold'] == self.fold_idx].index
            
            self.train_data = self.df.loc[train_idx].reset_index(drop=True)
            self.val_data = self.df.loc[val_idx].reset_index(drop=True)
        else:
            # For final training on all data
            if self.task == 'regression':
                self.train_data, self.val_data = train_test_split(
                    self.df, test_size=0.2, random_state=42
                )
            else:  # classification
                self.train_data, self.val_data = train_test_split(
                    self.df, test_size=0.2, random_state=42, stratify=self.df['Gas']
                )
        
        # Create datasets
        self.train_dataset = GasDataset(
            self.train_data, self.image_dir, transform=train_transform, preload=self.preload, task=self.task,
            standardize=self.standardize, mq8_mean=self.mq8_mean, mq8_std=self.mq8_std, 
            mq5_mean=self.mq5_mean, mq5_std=self.mq5_std
        )
        self.val_dataset = GasDataset(
            self.val_data, self.image_dir, transform=test_transform, preload=self.preload, task=self.task,
            standardize=self.standardize, mq8_mean=self.mq8_mean, mq8_std=self.mq8_std, 
            mq5_mean=self.mq5_mean, mq5_std=self.mq5_std
        )
        if self.test_df is not None:
            self.test_dataset = GasDataset(
                self.test_df, self.image_dir, transform=test_transform, is_test=True, preload=self.preload
            )
    
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

# Define softmax function for classification probabilities
def softmax(x, axis=1):
    # Subtracting the maximum for numerical stability
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Define the Lightning Module for the regression model
class GasRegressionModule(pl.LightningModule):
    def __init__(self, learning_rate=0.001, freeze_backbone=True, unfreeze_epoch=None,
                 standardize=False, mq8_mean=None, mq8_std=None, mq5_mean=None, mq5_std=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained ResNet50 model from timm
        self.backbone = timm.create_model('resnet50.a1h_in1k', pretrained=True)
        
        # Replace the final classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone weights if requested
        self.freeze_backbone = freeze_backbone
        self.unfreeze_epoch = unfreeze_epoch
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone weights frozen in regression model")
            if unfreeze_epoch is not None:
                print(f"Backbone will be unfrozen after epoch {unfreeze_epoch}")
        
        # 標準化に関する情報を保存
        self.standardize = standardize
        self.mq8_mean = mq8_mean
        self.mq8_std = mq8_std
        self.mq5_mean = mq5_mean
        self.mq5_std = mq5_std
        
        if standardize:
            print(f"Using standardized regression targets with:")
            print(f"  MQ8: mean={mq8_mean:.2f}, std={mq8_std:.2f}")
            print(f"  MQ5: mean={mq5_mean:.2f}, std={mq5_std:.2f}")
        
        # Create a regressor head with 2 outputs (MQ8, MQ5)
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.),
            nn.Linear(512, 2)
        )
        
        # Define loss function
        self.criterion = nn.MSELoss()
        
        # Track metrics
        self.train_preds, self.train_targets = [], []
        self.val_preds, self.val_targets = [], []

        self.train_mse = []
        self.val_mse = []
        self.train_mae = []
        self.val_mae = []
        self.train_r2 = []
        self.val_r2 = []
    
    def _unfreeze_backbone(self):
        """バックボーンの凍結を解除する"""
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze_backbone = False
            print("Backbone weights unfrozen in regression model")
    
    def _inverse_standardize(self, predictions):
        """標準化された予測値を元のスケールに戻す"""
        if not self.standardize or self.mq8_mean is None or self.mq8_std is None or self.mq5_mean is None or self.mq5_std is None:
            return predictions
        
        unstandardized_preds = predictions.clone()
        unstandardized_preds[:, 0] = predictions[:, 0] * self.mq8_std + self.mq8_mean
        unstandardized_preds[:, 1] = predictions[:, 1] * self.mq5_std + self.mq5_mean
        return unstandardized_preds
    
    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS_REGRESSION+1, eta_min=0.0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        
        # Track predictions and targets for metrics
        self.train_preds.append(predictions.detach())
        self.train_targets.append(y.detach())

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'preds': predictions, 'targets': y}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        
        # Track predictions and targets for metrics
        self.val_preds.append(predictions.detach())
        self.val_targets.append(y.detach())

        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'preds': predictions, 'targets': y}
    
    def predict_step(self, batch, batch_idx):
        # For test set, batch is just images without labels
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        
        predictions = self(x)
        return predictions
    
    def on_train_epoch_start(self):
        # 指定したエポック数に達したらバックボーンの凍結を解除
        if self.unfreeze_epoch is not None and self.current_epoch == self.unfreeze_epoch and self.freeze_backbone:
            self._unfreeze_backbone()
            
    def on_train_epoch_end(self):
        # Calculate metrics across the epoch
        all_preds = torch.cat(self.train_preds).cpu()
        all_targets = torch.cat(self.train_targets).cpu()
        
        # 予測値と真の値を元のスケールに戻す（必要な場合）
        if self.standardize and self.mq8_mean is not None and self.mq8_std is not None and self.mq5_mean is not None and self.mq5_std is not None:
            unstandardized_preds = self._inverse_standardize(all_preds)
            unstandardized_targets = self._inverse_standardize(all_targets)
            all_preds_np = unstandardized_preds.numpy()
            all_targets_np = unstandardized_targets.numpy()
        else:
            all_preds_np = all_preds.numpy()
            all_targets_np = all_targets.numpy()
        
        # Calculate MSE, MAE for each target (MQ8, MQ5)
        mse_mq8 = mean_squared_error(all_targets_np[:, 0], all_preds_np[:, 0])
        mse_mq5 = mean_squared_error(all_targets_np[:, 1], all_preds_np[:, 1])
        mae_mq8 = mean_absolute_error(all_targets_np[:, 0], all_preds_np[:, 0])
        mae_mq5 = mean_absolute_error(all_targets_np[:, 1], all_preds_np[:, 1])
        r2_mq8 = r2_score(all_targets_np[:, 0], all_preds_np[:, 0])
        r2_mq5 = r2_score(all_targets_np[:, 1], all_preds_np[:, 1])
        
        # Average metrics
        avg_mse = (mse_mq8 + mse_mq5) / 2
        avg_mae = (mae_mq8 + mae_mq5) / 2
        avg_r2 = (r2_mq8 + r2_mq5) / 2
        
        self.train_mse.append(avg_mse)
        self.train_mae.append(avg_mae)
        self.train_r2.append(avg_r2)
        
        self.log('train_mse', avg_mse, prog_bar=True)
        self.log('train_mae', avg_mae, prog_bar=True)
        self.log('train_r2', avg_r2, prog_bar=True)
        self.log('train_mse_mq8', mse_mq8)
        self.log('train_mse_mq5', mse_mq5)
        
        self.train_preds.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self):
        # Calculate metrics across the epoch
        all_preds = torch.cat(self.val_preds).cpu()
        all_targets = torch.cat(self.val_targets).cpu()
        
        # 予測値と真の値を元のスケールに戻す（必要な場合）
        if self.standardize and self.mq8_mean is not None and self.mq8_std is not None and self.mq5_mean is not None and self.mq5_std is not None:
            unstandardized_preds = self._inverse_standardize(all_preds)
            unstandardized_targets = self._inverse_standardize(all_targets)
            all_preds_np = unstandardized_preds.numpy()
            all_targets_np = unstandardized_targets.numpy()
        else:
            all_preds_np = all_preds.numpy()
            all_targets_np = all_targets.numpy()
        
        # Calculate MSE, MAE for each target (MQ8, MQ5)
        mse_mq8 = mean_squared_error(all_targets_np[:, 0], all_preds_np[:, 0])
        mse_mq5 = mean_squared_error(all_targets_np[:, 1], all_preds_np[:, 1])
        mae_mq8 = mean_absolute_error(all_targets_np[:, 0], all_preds_np[:, 0])
        mae_mq5 = mean_absolute_error(all_targets_np[:, 1], all_preds_np[:, 1])
        r2_mq8 = r2_score(all_targets_np[:, 0], all_preds_np[:, 0])
        r2_mq5 = r2_score(all_targets_np[:, 1], all_preds_np[:, 1])
        
        # Average metrics
        avg_mse = (mse_mq8 + mse_mq5) / 2
        avg_mae = (mae_mq8 + mae_mq5) / 2
        avg_r2 = (r2_mq8 + r2_mq5) / 2
        
        self.val_mse.append(avg_mse)
        self.val_mae.append(avg_mae)
        self.val_r2.append(avg_r2)
        
        self.log('val_mse', avg_mse, prog_bar=True)
        self.log('val_mae', avg_mae, prog_bar=True)
        self.log('val_r2', avg_r2, prog_bar=True)
        self.log('val_mse_mq8', mse_mq8)
        self.log('val_mse_mq5', mse_mq5)
        
        self.val_preds.clear()
        self.val_targets.clear()

        return {'val_mse': avg_mse, 'val_mae': avg_mae, 'val_r2': avg_r2}
    
    def get_metrics(self):
        return {
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'train_mae': self.train_mae,
            'val_mae': self.val_mae,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2
        }

# Function to get predictions
def get_regression_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(device)
            preds = model(x)
            
            # 標準化された予測値を元のスケールに戻す（必要な場合）
            if model.standardize and model.mq8_mean is not None and model.mq8_std is not None and model.mq5_mean is not None and model.mq5_std is not None:
                preds = model._inverse_standardize(preds)
                
            all_preds.append(preds.cpu().numpy())
    
    return np.vstack(all_preds)

# Function to get classification predictions with logits
def get_classification_predictions(model, dataloader, device):
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
    
    return np.vstack(all_logits)

# Dictionary to store best model paths for each fold
best_regression_models = {}

# Initialize OOF predictions arrays
oof_regression_preds = np.zeros((len(train_df), 2))
oof_classification_logits = np.zeros((len(train_df), len(label_map)))

# Initialize arrays for test predictions
if test_df is not None:
    test_regression_preds = np.zeros((len(test_df), 2))
    test_classification_logits = np.zeros((len(test_df), len(label_map)))

# Print training information
print("\n=== PHASE 1: REGRESSION TRAINING ===")
print(f"Number of folds: {NUM_FOLDS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE_REGRESSION}")
print(f"Number of epochs: {NUM_EPOCHS_REGRESSION}")
print(f"Preload images: {PRELOAD_IMAGES}")

# Collect metrics across folds for regression
cv_scores_regression = {'mse': [], 'mae': [], 'r2': []}

# Train regression models for each fold
for fold in range(NUM_FOLDS):
    print(f"\n{'='*50}\nTraining Regression Model - Fold {fold+1}/{NUM_FOLDS}\n{'='*50}")
    
    # Create data module for this fold
    data_module = GasDataModule(
        train_df, test_df, IMAGE_DIR, fold_idx=fold, batch_size=BATCH_SIZE, 
        preload=PRELOAD_IMAGES, task='regression',
        standardize=True, mq8_mean=mq8_mean, mq8_std=mq8_std, mq5_mean=mq5_mean, mq5_std=mq5_std
    )
    data_module.setup()
    
    # Get fold data sizes
    train_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index
    print(f"  Train: {len(train_idx)} samples, Validation: {len(val_idx)} samples")
    
    # Create model
    model = GasRegressionModule(
        learning_rate=LEARNING_RATE_REGRESSION, 
        freeze_backbone=True,
        unfreeze_epoch=UNFREEZE_EPOCH_REGRESSION,
        standardize=True, mq8_mean=mq8_mean, mq8_std=mq8_std, mq5_mean=mq5_mean, mq5_std=mq5_std
    )
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_DIR, 'checkpoints', f'fold_{fold}'),
        filename='gas-regressor-{epoch:02d}-{val_mse:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_mse',
        mode='min'
    )
    
    #early_stop_callback = EarlyStopping(
    #    monitor='val_mse',
    #    patience=5,
    #    verbose=True,
    #    mode='min'
    #)
    
    # Create logger
    logger = TensorBoardLogger(os.path.join(OUTPUT_DIR, 'lightning_logs'), name=f'gas_regressor_fold_{fold}')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS_REGRESSION,
        callbacks=[checkpoint_callback],# early_stop_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        log_every_n_steps=10,
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
    print(f"Saved regression model weights to {model_save_path}")
    
    # Store the best model path for this fold (for use in classification phase)
    best_regression_models[fold] = model_save_path
    
    # Load best model for predictions
    best_model = GasRegressionModule.load_from_checkpoint(best_model_path)
    best_model.eval()
    
    # Get validation data for this fold
    val_data, val_dataset = data_module.get_val_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Make OOF predictions for this fold
    val_preds_fold = get_regression_predictions(best_model, val_dataloader, best_model.device)
    oof_regression_preds[val_idx] = val_preds_fold
    
    # Calculate metrics for this fold
    y_val_mq8 = val_data['MQ8'].values
    y_val_mq5 = val_data['MQ5'].values
    y_val = np.column_stack((y_val_mq8, y_val_mq5))
    
    mse = mean_squared_error(y_val, val_preds_fold)
    mae = mean_absolute_error(y_val, val_preds_fold)
    r2 = r2_score(y_val, val_preds_fold)
    
    cv_scores_regression['mse'].append(mse)
    cv_scores_regression['mae'].append(mae)
    cv_scores_regression['r2'].append(r2)
    
    print(f"  Fold {fold + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Calculate and print metrics for each target
    mse_mq8 = mean_squared_error(y_val[:, 0], val_preds_fold[:, 0])
    mse_mq5 = mean_squared_error(y_val[:, 1], val_preds_fold[:, 1])
    mae_mq8 = mean_absolute_error(y_val[:, 0], val_preds_fold[:, 0])
    mae_mq5 = mean_absolute_error(y_val[:, 1], val_preds_fold[:, 1])
    r2_mq8 = r2_score(y_val[:, 0], val_preds_fold[:, 0])
    r2_mq5 = r2_score(y_val[:, 1], val_preds_fold[:, 1])
    
    print(f"  MQ8 - MSE: {mse_mq8:.4f}, MAE: {mae_mq8:.4f}, R²: {r2_mq8:.4f}")
    print(f"  MQ5 - MSE: {mse_mq5:.4f}, MAE: {mae_mq5:.4f}, R²: {r2_mq5:.4f}")
    
    # Plot actual vs. predicted for this fold
    plt.figure(figsize=(15, 6))
    
    # Plot for MQ8
    plt.subplot(1, 2, 1)
    plt.scatter(y_val[:, 0], val_preds_fold[:, 0], alpha=0.5)
    plt.plot([y_val[:, 0].min(), y_val[:, 0].max()], [y_val[:, 0].min(), y_val[:, 0].max()], 'r--')
    plt.xlabel('Actual MQ8')
    plt.ylabel('Predicted MQ8')
    plt.title(f'Fold {fold + 1} - MQ8 Predictions (R² = {r2_mq8:.4f})')
    
    # Plot for MQ5
    plt.subplot(1, 2, 2)
    plt.scatter(y_val[:, 1], val_preds_fold[:, 1], alpha=0.5)
    plt.plot([y_val[:, 1].min(), y_val[:, 1].max()], [y_val[:, 1].min(), y_val[:, 1].max()], 'r--')
    plt.xlabel('Actual MQ5')
    plt.ylabel('Predicted MQ5')
    plt.title(f'Fold {fold + 1} - MQ5 Predictions (R² = {r2_mq5:.4f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_fold_{fold}.png'))
    plt.close()
    
    # Make test predictions for this fold if test data exists
    if test_df is not None:
        test_dataloader = data_module.test_dataloader()
        test_preds_fold = get_regression_predictions(best_model, test_dataloader, best_model.device)
        test_regression_preds += test_preds_fold / NUM_FOLDS

# Create OOF dataframe with predictions for regression
oof_regression_df = train_df.copy()
oof_regression_df['predicted_MQ8'] = oof_regression_preds[:, 0]
oof_regression_df['predicted_MQ5'] = oof_regression_preds[:, 1]

# Calculate OOF metrics for regression
y_true_mq8 = oof_regression_df['MQ8'].values
y_true_mq5 = oof_regression_df['MQ5'].values
y_true = np.column_stack((y_true_mq8, y_true_mq5))

oof_mse = mean_squared_error(y_true, oof_regression_preds)
oof_mae = mean_absolute_error(y_true, oof_regression_preds)
oof_r2 = r2_score(y_true, oof_regression_preds)

oof_mse_mq8 = mean_squared_error(y_true[:, 0], oof_regression_preds[:, 0])
oof_mse_mq5 = mean_squared_error(y_true[:, 1], oof_regression_preds[:, 1])
oof_mae_mq8 = mean_absolute_error(y_true[:, 0], oof_regression_preds[:, 0])
oof_mae_mq5 = mean_absolute_error(y_true[:, 1], oof_regression_preds[:, 1])
oof_r2_mq8 = r2_score(y_true[:, 0], oof_regression_preds[:, 0])
oof_r2_mq5 = r2_score(y_true[:, 1], oof_regression_preds[:, 1])

print("\nRegression Out-of-fold prediction results:")
print(f"OOF MSE: {oof_mse:.4f}")
print(f"OOF MAE: {oof_mae:.4f}")
print(f"OOF R²: {oof_r2:.4f}")

print("\nIndividual target metrics:")
print(f"MQ8 - MSE: {oof_mse_mq8:.4f}, MAE: {oof_mae_mq8:.4f}, R²: {oof_r2_mq8:.4f}")
print(f"MQ5 - MSE: {oof_mse_mq5:.4f}, MAE: {oof_mae_mq5:.4f}, R²: {oof_r2_mq5:.4f}")

# Save OOF predictions (only for internal analysis, not actual output)
# Comment out to avoid saving regression files
# oof_regression_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_oof_predictions.csv'), index=False)
# print(f"OOF regression predictions saved to {os.path.join(OUTPUT_DIR, 'regression_oof_predictions.csv')}")

print("\nRegression Cross-validation results:")
print(f"Mean MSE: {np.mean(cv_scores_regression['mse']):.4f} ± {np.std(cv_scores_regression['mse']):.4f}")
print(f"Mean MAE: {np.mean(cv_scores_regression['mae']):.4f} ± {np.std(cv_scores_regression['mae']):.4f}")
print(f"Mean R²: {np.mean(cv_scores_regression['r2']):.4f} ± {np.std(cv_scores_regression['r2']):.4f}")

# Plot overall OOF predictions for regression
plt.figure(figsize=(15, 6))

# Plot for MQ8
plt.subplot(1, 2, 1)
plt.scatter(y_true[:, 0], oof_regression_preds[:, 0], alpha=0.5)
plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], [y_true[:, 0].min(), y_true[:, 0].max()], 'r--')
plt.xlabel('Actual MQ8')
plt.ylabel('Predicted MQ8')
plt.title(f'Overall OOF MQ8 Predictions (R² = {oof_r2_mq8:.4f})')

# Plot for MQ5
plt.subplot(1, 2, 2)
plt.scatter(y_true[:, 1], oof_regression_preds[:, 1], alpha=0.5)
plt.plot([y_true[:, 1].min(), y_true[:, 1].max()], [y_true[:, 1].min(), y_true[:, 1].max()], 'r--')
plt.xlabel('Actual MQ5')
plt.ylabel('Predicted MQ5')
plt.title(f'Overall OOF MQ5 Predictions (R² = {oof_r2_mq5:.4f})')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overall_regression.png'))
plt.close()

# Save regression test predictions if test data exists - commented out to avoid saving regression files
if test_df is not None:
    test_regression_df = test_df.copy()
    test_regression_df['predicted_MQ8'] = test_regression_preds[:, 0]
    test_regression_df['predicted_MQ5'] = test_regression_preds[:, 1]
    # test_regression_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_test_predictions.csv'), index=False)
    # print(f"Test regression predictions saved to {os.path.join(OUTPUT_DIR, 'regression_test_predictions.csv')}")

print("\n=== PHASE 2: CLASSIFICATION TRAINING ===")
print(f"Number of folds: {NUM_FOLDS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE_CLASSIFICATION}")
print(f"Number of epochs: {NUM_EPOCHS_CLASSIFICATION}")
print(f"Using pre-trained regression backbone: True")

# Collect metrics across folds for classification
cv_scores_classification = {'accuracy': [], 'f1': []}

# Define the Lightning Module for the classification model using pre-trained backbone
class GasClassifierModule(pl.LightningModule):
    def __init__(self, pretrained_backbone_path=None, num_classes=4, learning_rate=0.001, freeze_backbone=True, unfreeze_epoch=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained ResNet50 model from timm
        self.backbone = timm.create_model('resnet50.a1h_in1k', pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Load pre-trained weights from regression model if provided
        if pretrained_backbone_path:
            # Load only the backbone weights
            state_dict = torch.load(pretrained_backbone_path)
            # Filter out the regressor weights, keep only backbone
            backbone_state_dict = {k: v for k, v in state_dict.items() if k.startswith('backbone.')}
            # Load the weights
            self.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in backbone_state_dict.items()})
            print(f"Loaded pre-trained backbone weights from {pretrained_backbone_path}")
        
        # Freeze backbone weights if requested
        self.freeze_backbone = freeze_backbone
        self.unfreeze_epoch = unfreeze_epoch
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone weights frozen")
            if unfreeze_epoch is not None:
                print(f"Backbone will be unfrozen after epoch {unfreeze_epoch}")
        
        # Create a classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
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
    
    def _unfreeze_backbone(self):
        """バックボーンの凍結を解除する"""
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze_backbone = False
            print("Backbone weights unfrozen in classification model")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS_CLASSIFICATION+1, eta_min=0.0)
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
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return preds, probs
    
    def on_train_epoch_start(self):
        # 指定したエポック数に達したらバックボーンの凍結を解除
        if self.unfreeze_epoch is not None and self.current_epoch == self.unfreeze_epoch and self.freeze_backbone:
            self._unfreeze_backbone()
    
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

# Train classification models for each fold
for fold in range(NUM_FOLDS):
    print(f"\n{'='*50}\nTraining Classification Model - Fold {fold+1}/{NUM_FOLDS}\n{'='*50}")
    
    # Create data module for this fold
    data_module = GasDataModule(train_df, test_df, IMAGE_DIR, fold_idx=fold, batch_size=BATCH_SIZE, preload=PRELOAD_IMAGES, task='classification')
    data_module.setup()
    
    # Get fold data sizes
    train_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index
    print(f"  Train: {len(train_idx)} samples, Validation: {len(val_idx)} samples")
    
    # Get regression model weights for this fold
    regression_weights_path = best_regression_models[fold]
    
    # Create classification model with pre-trained backbone
    model = GasClassifierModule(
        pretrained_backbone_path=regression_weights_path,
        num_classes=len(label_map),
        learning_rate=LEARNING_RATE_CLASSIFICATION,
        freeze_backbone=True,
        unfreeze_epoch=UNFREEZE_EPOCH_CLASSIFICATION
    )
    
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
    #    patience=3,
    #    verbose=True,
    #    mode='max'
    #)
    
    # Create logger
    logger = TensorBoardLogger(os.path.join(OUTPUT_DIR, 'lightning_logs'), name=f'gas_classifier_fold_{fold}')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS_CLASSIFICATION,
        callbacks=[checkpoint_callback],# early_stop_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        log_every_n_steps=10,
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
    print(f"Saved classification model weights to {model_save_path}")
    
    # Load best model for predictions
    best_model = GasClassifierModule.load_from_checkpoint(best_model_path)
    best_model.eval()
    
    # Get validation data for this fold
    val_data, val_dataset = data_module.get_val_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Make OOF predictions for this fold (with logits)
    val_logits_fold = get_classification_predictions(best_model, val_dataloader, best_model.device)
    oof_classification_logits[val_idx] = val_logits_fold
    
    # Convert logits to probabilities for metrics calculation
    val_probs = softmax(val_logits_fold, axis=1)
    val_pred_class = np.argmax(val_probs, axis=1)
    y_val = val_data['Gas'].map(label_map).values
    
    accuracy = accuracy_score(y_val, val_pred_class)
    f1 = f1_score(y_val, val_pred_class, average='weighted')
    
    cv_scores_classification['accuracy'].append(accuracy)
    cv_scores_classification['f1'].append(f1)
    
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
    plt.savefig(os.path.join(OUTPUT_DIR, f'fold{fold}_confusion.png'))
    plt.close()
    
    # Make test predictions for this fold if test data exists
    if test_df is not None:
        test_dataloader = data_module.test_dataloader()
        test_logits_fold = get_classification_predictions(best_model, test_dataloader, best_model.device)
        test_classification_logits += test_logits_fold / NUM_FOLDS

# Convert averaged logits to probabilities
oof_classification_preds = softmax(oof_classification_logits, axis=1)

# Create OOF dataframe with probabilities
oof_classification_df = train_df.copy()
for i, label in enumerate(label_map.keys()):
    oof_classification_df[f'prob_{label}'] = oof_classification_preds[:, i]

# Add predicted class
oof_classification_df['predicted_class_idx'] = np.argmax(oof_classification_preds, axis=1)
oof_classification_df['predicted_class'] = oof_classification_df['predicted_class_idx'].map(inv_label_map)

# Calculate OOF metrics
oof_pred_class = np.argmax(oof_classification_preds, axis=1)
y_true = oof_classification_df['Gas'].map(label_map).values

oof_accuracy = accuracy_score(y_true, oof_pred_class)
oof_f1 = f1_score(y_true, oof_pred_class, average='weighted')

print("\nClassification Out-of-fold prediction results:")
print(f"OOF Accuracy: {oof_accuracy:.4f}")
print(f"OOF F1 Score: {oof_f1:.4f}")

# Save OOF predictions
oof_classification_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_predictions_probs.csv'), index=False)
print(f"OOF classification predictions saved to {os.path.join(OUTPUT_DIR, 'oof_predictions_probs.csv')}")

print("\nClassification Cross-validation results:")
print(f"Mean Accuracy: {np.mean(cv_scores_classification['accuracy']):.4f} ± {np.std(cv_scores_classification['accuracy']):.4f}")
print(f"Mean F1 Score: {np.mean(cv_scores_classification['f1']):.4f} ± {np.std(cv_scores_classification['f1']):.4f}")

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
    plt.hist(oof_classification_preds[:, i], bins=50)
    plt.title(f'OOF Probability Distribution - {label}')
    plt.xlabel('Probability')
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'oof_distribution.png'))
plt.close()

# Create test submission with final ensemble predictions if test data exists
if test_df is not None:
    # Convert averaged test logits to probabilities
    test_classification_preds = softmax(test_classification_logits, axis=1)
    
    test_pred_classes = np.argmax(test_classification_preds, axis=1)
    test_pred_labels = [inv_label_map[pred] for pred in test_pred_classes]

    # Create submission file
    submission = pd.DataFrame({
        'index': range(len(test_pred_labels)),
        'Gas': test_pred_labels
    })

    # Add probability columns to submission
    for i, label in enumerate(label_map.keys()):
        submission[f'prob_{label}'] = test_classification_preds[:, i]

    submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"Classification submission file saved to {os.path.join(OUTPUT_DIR, 'submission.csv')}")
    
    # Create and save test_predictions_probs.csv with more details
    test_classification_df = test_df.copy()
    # Add probability columns
    for i, label in enumerate(label_map.keys()):
        test_classification_df[f'prob_{label}'] = test_classification_preds[:, i]
    
    # Add predicted class columns
    test_classification_df['predicted_class_idx'] = test_pred_classes
    test_classification_df['predicted_class'] = test_pred_labels
    
    # Save test predictions with probabilities
    test_classification_df.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv'), index=False)
    print(f"Test classification predictions saved to {os.path.join(OUTPUT_DIR, 'test_predictions_probs.csv')}")

print("\nTwo-phase training (regression + classification) completed successfully!")
