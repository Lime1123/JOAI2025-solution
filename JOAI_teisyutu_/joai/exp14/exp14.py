import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Set random seeds for reproducibility
pl.seed_everything(42)

# Define paths
DATA_DIR = 'data'  # Update this path if your data is located elsewhere
OUTPUT_DIR = 'joai/exp14'  # Directory to save models and outputs

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)

# Define constants
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
NUM_FOLDS = 5  # Number of folds for K-Fold CV
MAX_LEN = 128  # Maximum length of token sequences

# Load data
print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_folds_temp.csv'))
test_df = None

# Check if test.csv exists
test_path = os.path.join(DATA_DIR, 'test.csv')
if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")

print(f"Training data shape: {train_df.shape}")

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

# Create a dataset class for text
class CaptionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, is_test=False):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.df.iloc[idx]['Caption']
        
        # Tokenize the caption
        encoding = self.tokenizer(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Get the input ids and attention mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        if self.is_test:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            label = label_map[self.df.iloc[idx]['Gas']]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(label, dtype=torch.long)
            }

def softmax(x, axis=1):
    # Subtracting the maximum for numerical stability
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Lightning Data Module for K-Fold CV
class CaptionDataModule(pl.LightningDataModule):
    def __init__(self, df, test_df, tokenizer, fold_idx=None, batch_size=16, max_len=128):
        super().__init__()
        self.df = df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.max_len = max_len
    
    def setup(self, stage=None):
        if self.fold_idx is not None:
            # For K-Fold CV
            train_idx = self.df[self.df['fold'] != self.fold_idx].index
            val_idx = self.df[self.df['fold'] == self.fold_idx].index
            
            self.train_data = self.df.loc[train_idx].reset_index(drop=True)
            self.val_data = self.df.loc[val_idx].reset_index(drop=True)
        else:
            # For final training on all data
            train_idx = self.df[self.df['fold'] != 0].index
            val_idx = self.df[self.df['fold'] == 0].index
            
            self.train_data = self.df.loc[train_idx].reset_index(drop=True)
            self.val_data = self.df.loc[val_idx].reset_index(drop=True)
        
        # Create datasets
        self.train_dataset = CaptionDataset(self.train_data, self.tokenizer, self.max_len)
        self.val_dataset = CaptionDataset(self.val_data, self.tokenizer, self.max_len)
        if self.test_df is not None:
            self.test_dataset = CaptionDataset(self.test_df, self.tokenizer, self.max_len, is_test=True)
    
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

# Define the Lightning Module for the RoBERTa model
class RoBERTaClassifierModule(pl.LightningModule):
    def __init__(self, model_name='FacebookAI/roberta-base', num_classes=4, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained RoBERTa model
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Add a classifier on top
        hidden_size = self.roberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
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
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token representation
        return self.classifier(pooled_output)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        # Define scheduler
        total_steps = NUM_EPOCHS * (len(train_df) // BATCH_SIZE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.train_preds.append(preds)
        self.train_targets.append(labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'preds': preds, 'targets': labels}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.val_preds.append(preds)
        self.val_targets.append(labels)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        # Return probabilities for OOF predictions
        return {'val_loss': loss, 'preds': preds, 'targets': labels, 'logits': logits}
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        logits = self(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
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

# Function to get predictions with logits
def get_predictions(model, dataloader, device):
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu().numpy())
    
    return np.vstack(all_logits)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

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
print(f"Max sequence length: {MAX_LEN}")

# Collect metrics across folds
cv_scores = {'accuracy': [], 'f1': []}

# Train models for each fold
for fold in range(NUM_FOLDS):
    print(f"\n{'='*50}\nTraining Fold {fold+1}/{NUM_FOLDS}\n{'='*50}")
    
    # Create data module for this fold
    data_module = CaptionDataModule(
        train_df, test_df, tokenizer, 
        fold_idx=fold, batch_size=BATCH_SIZE, max_len=MAX_LEN
    )
    data_module.setup()
    
    # Get fold data sizes
    train_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index
    print(f"  Train: {len(train_idx)} samples, Validation: {len(val_idx)} samples")
    
    # Create model
    model = RoBERTaClassifierModule(
        model_name='FacebookAI/roberta-base',
        num_classes=4,
        learning_rate=LEARNING_RATE
    )
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_DIR, 'checkpoints', f'fold_{fold}'),
        filename='roberta-classifier-{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_f1',
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=3,
        verbose=True,
        mode='max'
    )
    
    # Create logger
    logger = TensorBoardLogger(os.path.join(OUTPUT_DIR, 'lightning_logs'), name=f'roberta_classifier_fold_{fold}')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
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
    #model_save_path = os.path.join(OUTPUT_DIR, 'models', f'fold_{fold}_best_model.pth')
    #torch.save(model.state_dict(), model_save_path)
    #print(f"Saved model weights to {model_save_path}")
    
    # Load best model for predictions
    best_model = RoBERTaClassifierModule.load_from_checkpoint(best_model_path)
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
