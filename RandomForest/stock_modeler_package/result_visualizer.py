import matplotlib.pyplot as plt

class ResultVisualizer:
    def __init__(self, data_handler, returns_dict: dict):
        """
        data_handler: your DataHandler instance
        returns_dict: dict of label -> DataFrame with 'monthly_return' & 'value'
        """
        self.data_handler = data_handler
        self.returns_dict = returns_dict

    def plot_monthly_returns(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, df in self.returns_dict.items():
            print("DataFrame columns:", df.columns.tolist())
            print("First few rows:\n", df.head())
            ax.plot(df.index, df['monthly_return'], label=label)
            
        ax.set_title('Average Monthly Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Monthly Return')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return fig

    def plot_portfolio_value(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, df in self.returns_dict.items():
            ax.plot(df.index, df['portfolio_value'], label=label)
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return fig

    def show_all(self):
        fig1 = self.plot_monthly_returns()
        plt.show()
        fig2 = self.plot_portfolio_value()
        plt.show()
        return fig1