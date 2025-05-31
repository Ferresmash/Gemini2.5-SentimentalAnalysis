import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioSimulator:
    
    def __init__(self,
                 predictions: pd.DataFrame,
                 actual_returns: pd.DataFrame):
    # 1) copy + parse dates
        preds = predictions.copy()
        preds.index = pd.to_datetime(preds.index)

        rets = actual_returns.copy()
        rets.index = pd.to_datetime(rets.index)

        # 2) strip out any "target_" prefix and "_returns_t+1" suffix
        #    from **both** sets of column names:
        strip_re = r"^target_|_returns?_t\+1$"
        preds.columns = preds.columns.str.replace(strip_re, "", regex=True)
        rets.columns  = rets.columns.str.replace(strip_re, "", regex=True)

        # 3) now intersect dates
        common = preds.index.intersection(rets.index)
        if common.empty:
            raise ValueError("No overlapping dates between predictions and returns.")

        # 4) slice to the common dates
        self.predictions = preds.loc[common]
        self.returns     = rets.loc[common]
        self.dates       = common

        # 5) sanity check that columns now match
        missing = set(self.predictions.columns) - set(self.returns.columns)
        if missing:
            raise ValueError(f"No return series for predicted tickers: {missing}")

        self.tickers = list(self.predictions.columns)
        self.index_weights = None
        print(f"Initialized with tickers: {self.tickers}")
        
    def load_index_weights(
        self,
        csv_path: str = r'C:\Users\ohrnf\Examensarbete-2025\Gemini2.5-SentimentalAnalysis\RandomForest\index_weights.csv',
        ticker_col: str = 'Ticker',
        weight_col: str = 'Weight'
    ):
        # 1) Read CSV
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8', decimal='.')
        print(f"Loaded index weights from {csv_path} with shape {df.shape}")

        # 2) Clean column names
        df.columns = df.columns.str.strip()
        ticker_col_clean = ticker_col.strip()
        weight_col_clean = weight_col.strip()
        if ticker_col_clean not in df.columns or weight_col_clean not in df.columns:
            raise KeyError(
                f"Expected columns '{ticker_col_clean}' and '{weight_col_clean}' in weights file, "
                f"found {df.columns.tolist()}"
            )

        # 3) Coerce to float
        df[weight_col_clean] = (
            df[weight_col_clean]
            .astype(str)
            .str.strip()
            .astype(float)
        )

        # 4) Build the Series and normalize so weights sum to 1
        raw = df.set_index(ticker_col_clean)[weight_col_clean]
        normalized = raw / raw.sum()
        print(f"Normalized index weights (sum = {normalized.sum():.4f}):\n{normalized}")

        # 5) Assign
        self.index_weights = normalized

    def simulate_strategy(
        self,
        top_k: int = 6,
        bottom_k: int = 6,
        initial_capital: float = 1000.0,
    ) -> pd.DataFrame:
        """
        Runs the long-short transfer strategy.
        Returns a DataFrame indexed by date with columns:
          - portfolio_value: cumulative value over time
          - monthly_return: simple return each period
        """
        N = len(self.tickers)
        if N <= top_k + bottom_k:
            raise ValueError("Need more assets than top_k + bottom_k to transfer capital.")

        base_w = 1.0 / N
        transfer_amt = bottom_k * base_w
        w_win = base_w + transfer_amt / top_k

        # storage
        values = []
        returns = []

        prev_cap = initial_capital
        cap = initial_capital

        for date in self.dates:
            preds = self.predictions.loc[date]
            rets  = self.returns.loc[date]

            # rank and choose
            ranked = preds.sort_values()
            losers  = ranked.index[:bottom_k]
            winners = ranked.index[-top_k:]

            # set weights
            w = pd.Series(base_w, index=self.tickers)
            w.loc[losers]  = 0.0
            w.loc[winners] = w_win

            # compute new capital
            cap = (cap * w * (1.0 + rets)).sum()

            # record
            values.append(cap)
            returns.append((cap / prev_cap) - 1.0)

            prev_cap = cap
            
        print("returns",returns)    
        print("values",values)

        return pd.DataFrame({
            'portfolio_value': values,
            'monthly_return': returns
        }, index=self.dates)


    def simulate_index(
        self,
        initial_capital: float = 1000.0,
    ) -> pd.DataFrame:
        """
        Runs passive buy-and-hold index simulation.
        Must call load_index_weights() first.
        Returns a DataFrame indexed by date with columns:
          - portfolio_value: cumulative value over time
          - monthly_return: simple return each period
        """
        if self.index_weights is None:
            self.load_index_weights()
        

        # verify ticker coverage
        missing = self.index_weights.index.difference(self.returns.columns)
        if not missing.empty:
            raise ValueError(f"Tickers in index weights missing from returns: {missing.tolist()}")

        # initial positions in currency units
        pos = initial_capital * self.index_weights

        values  = []
        returns = []

        prev_value = initial_capital

        for date in self.dates:
            rets = self.returns.loc[date]
            pos  = pos * (1.0 + rets)
            total = pos.sum()

            values.append(total)
            returns.append((total / prev_value) - 1.0)

            prev_value = total
            
        print("returns",returns)    
        print("values",values)

        return pd.DataFrame({
            'portfolio_value': values,
            'monthly_return':  returns
        }, index=self.dates)