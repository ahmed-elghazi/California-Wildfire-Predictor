from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

class DataPreprocessorAndSplitter:
    def __init__(self, df, target_col, drop_cols=None, cat_cols=None, num_cols=None):
        self.df = df.dropna()
        self.target_col = target_col
        self.drop_cols = drop_cols if drop_cols else []
        self.cat_cols = cat_cols if cat_cols else []
        self.num_cols = num_cols if num_cols else []
        self.splits = {}
        self.pipeline = None

    def _build_pipeline(self):
        transformers = []
        if self.num_cols:
            transformers.append(('num', StandardScaler(), self.num_cols))
        if self.cat_cols:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_cols))
        return ColumnTransformer(transformers)

    def stratified_split(self, test_size=0.2, val_size=0.25, random_state=42):
        X = self.df.drop(columns=[self.target_col] + self.drop_cols)
        y = self.df[self.target_col]

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size,
                                                        stratify=y_temp, random_state=random_state)

        self.pipeline = self._build_pipeline()
        X_train = self.pipeline.fit_transform(X_train)
        X_val = self.pipeline.transform(X_val)
        X_test = self.pipeline.transform(X_test)

        self.splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
        }
        return self.splits

    def class_distribution(self):
        return {k: y.value_counts(normalize=True) for k, (_, y) in self.splits.items()}
