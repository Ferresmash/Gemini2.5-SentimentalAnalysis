import pandas as pd
import os

class DataHandler:
    def __init__(self, filepath, date_col='Date'):
        self.filepath = filepath
        self.date_col = date_col
        self.X_all_features = None
        self.Y_all_targets = None
        self.full_target_names = None
        self.current_return_cols = None
        self.current_sentiment_cols = None
        self.lags_used = None

    def load_and_prepare_data(self):
        # Load CSV
        if not os.path.isfile(self.filepath + '/t_data.csv'):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        df = pd.read_csv(self.filepath+ '/t_data.csv', delimiter=';', decimal='.', encoding='utf-8-sig')
        
        # Parse date column and set index
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df.set_index(self.date_col, inplace=True)
        else:
            # assume first column is date
            idx = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            if pd.isna(idx).all():
                raise ValueError("Could not parse any dates from first column or specified date_col")
            df.index = idx
            df = df.iloc[:, 1:]

        # Identify return and sentiment columns
        self.current_return_cols = sorted([c for c in df.columns if c.endswith('_returns')])
        self.current_sentiment_cols = sorted([c for c in df.columns if c.endswith('_sentiment')])

        # Create lagged features
        self.lags_used = [1, 2, 3]
        lagged = []
        for col in self.current_return_cols + self.current_sentiment_cols:
            for lag in self.lags_used:
                lagged_col = f"{col}_lag{lag}"
                lagged.append(df[col].shift(lag).rename(lagged_col))
        if lagged:
            df = pd.concat([df] + lagged, axis=1)

        # Build feature matrix X
        feature_cols = self.current_return_cols + self.current_sentiment_cols + \
                       [c for c in df.columns if '_lag' in c]
        X = df[feature_cols]

        # Build target matrix Y (shift returns -1)
        Y = pd.concat([
            df[ret].shift(-1).rename(f"target_{ret}_t+1")
            for ret in self.current_return_cols
        ], axis=1)

        # Drop rows with any NaN across X or Y (ensures alignment)
        combined = pd.concat([X, Y], axis=1).dropna()

        # Split back
        self.X_all_features = combined[feature_cols]
        self.Y_all_targets = combined[[c for c in combined.columns if c.startswith('target_')]]
        self.full_target_names = list(self.Y_all_targets.columns)

        print(f"[DataHandler] Features shape: {self.X_all_features.shape}, "
              f"Targets shape: {self.Y_all_targets.shape}")
        return True

    def getFeaturesFromStock(self, stock_name, model_type='full'):
        # stock_name like 'target_<col>_t+1', strip to base
        base = stock_name.replace('target_', '').replace('_returns_t+1', '')
        # select base returns and sentiment and their lags
        ret_cols = [c for c in self.current_return_cols if c.startswith(base) and '_returns' in c]
        sent_cols = [c for c in self.current_sentiment_cols if c.startswith(base) and '_sentiment' in c]
        lag_ret = [c for c in self.X_all_features.columns if c.startswith(base) and '_lag' in c and '_returns' in c]
        lag_sent = [c for c in self.X_all_features.columns if c.startswith(base) and '_lag' in c and '_sentiment' in c]

        if model_type == 'full':
            cols = ret_cols + sent_cols + lag_ret + lag_sent
        elif model_type == 'returns_only':
            cols = ret_cols + lag_ret
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        return self.X_all_features[cols]
