import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Load the training data
df = pd.read_csv('data/train.csv')

# Print information about the dataset
print(f"Training data shape: {df.shape}")
print(f"Target distribution: \n{df['Gas'].value_counts()}")

# Number of folds
n_splits = 5

# Initialize the StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Add a fold column initialized with -1
df['fold'] = -1

# Fill in fold numbers
for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['Gas'])):
    df.loc[val_idx, 'fold'] = fold

# Verify the fold distribution
print("\nFold distribution:")
print(df['fold'].value_counts())

# Verify target distribution in each fold
print("\nTarget distribution in each fold:")
for fold in range(n_splits):
    fold_df = df[df['fold'] == fold]
    print(f"Fold {fold}:")
    print(fold_df['Gas'].value_counts())
    print(f"Total: {len(fold_df)}")
    print()

# Save the updated dataset
df.to_csv('data/train_with_folds.csv', index=False)
print("Added fold column and saved to 'data/train_with_folds.csv'")
