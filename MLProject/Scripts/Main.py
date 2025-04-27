import pandas as pd
import joblib
import numpy as np
from DataSplitter import DataPreprocessorAndSplitter
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("all_conditions.csv")

# 2. Define which columns to drop and which are numeric
drop_cols = ['Stn Name', 'CIMIS Region', 'Date']  # Drop these
target_col = 'Target'

numeric_cols = [
    'Stn Id', 'ETo (in)', 'Precip (in)', 'Sol Rad (Ly/day)',
    'Avg Vap Pres (mBars)', 'Max Air Temp (F)', 'Min Air Temp (F)',
    'Avg Air Temp (F)', 'Max Rel Hum (%)', 'Min Rel Hum (%)',
    'Avg Rel Hum (%)', 'Dew Point (F)', 'Avg Wind Speed (mph)',
    'Wind Run (miles)', 'Avg Soil Temp (F)'
]

# 3. Use the preprocessing and splitter class
splitter = DataPreprocessorAndSplitter(
    df=df,
    target_col=target_col,
    drop_cols=drop_cols,
    num_cols=numeric_cols
)

splits = splitter.stratified_split()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = splitter.splits['train'], splitter.splits['val'], splitter.splits['test']

# Apply undersampling to training set only
undersample = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)

# Optional: Print to verify
import numpy as np
print("Original train distribution:", np.bincount(y_train))
print("Resampled train distribution:", np.bincount(y_train_resampled))

# Save the new resampled balanced training data
np.save('data/X_train.npy', X_train_resampled)
np.save('data/y_train.npy', y_train_resampled)

# Save val/test normally
np.save('data/X_val.npy', X_val)
np.save('data/y_val.npy', y_val)
np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test)

# Save the preprocessing pipeline
joblib.dump(splitter.pipeline, 'models/preprocessing_pipeline.joblib')
distributions = splitter.class_distribution()

# 4. Optional: Print to verify class balance
for name, dist in distributions.items():
    print(f"{name.upper()}:\n{dist}\n")


