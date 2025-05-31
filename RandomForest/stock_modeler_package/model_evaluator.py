import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import linregress

class ModelEvaluator:
    def __init__(self, data_handler, tuned_params, tuned_params_returns_only, tscv_eval_splits=3):
        self.data_handler = data_handler
        self.tuned_params = tuned_params
        self.tuned_params_returns_only = tuned_params_returns_only
        self.tscv_eval = TimeSeriesSplit(n_splits=tscv_eval_splits)
        self.results_dfs = {}
        self.overall_r2_scores = []


    def evaluate_models(self, model_type="full"):
        Y_all = self.data_handler.Y_all_targets.copy()
        tscv = self.tscv_eval

        cv_preds = pd.DataFrame(index=Y_all.index)
        results = {}

        print(f"\n--- Evaluating models ({model_type}): CV metrics + full-sample preds ---")

        for target in Y_all.columns:
            X = self.data_handler.getFeaturesFromStock(target, model_type=model_type)
            y = Y_all[target]
            params = (self.tuned_params if model_type == "full" else self.tuned_params_returns_only)[target]

            series_cv = pd.Series(index=Y_all.index, dtype=float)
            scores = {'mse': [], 'mae': [], 'r2': []}

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model = RandomForestRegressor(**params, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                dates = X_test.index
                series_cv.loc[dates] = y_pred

                # record metrics
                scores['mse'].append(mean_squared_error(y_test, y_pred))
                scores['mae'].append(mean_absolute_error(y_test, y_pred))
                scores['r2'].append(r2_score(y_test, y_pred))


            results[target] = {
                'mse_mean': np.mean(scores['mse']), 'mse_std': np.std(scores['mse']),
                'mae_mean': np.mean(scores['mae']), 'mae_std': np.std(scores['mae']),
                'r2_mean': np.mean(scores['r2']), 'r2_std': np.std(scores['r2'])
            }
            
            print(f"  [{target}] CV averages: R2={results[target]['r2_mean']:.4f}±{results[target]['r2_std']:.4f}")


            cv_preds[target] = series_cv

        cv_preds.dropna(how='all', inplace=True)
        cv_preds.index = pd.to_datetime(cv_preds.index)

        self.results_dfs[model_type] = pd.DataFrame(results)
        self.overall_r2_scores.append({model_type: [r['r2_mean'] for r in results.values()]})


        print("\n--- Evaluation complete. CV preds generated. ---")
        return cv_preds

    def t_test (self, predictions: pd.DataFrame, predictions_returns_only: pd.DataFrame):
        """
        Perform a t-test to compare the performance of two sets of predictions,
        based on the mean prediction across all commonly available stocks at each time point.
        """

        common_idx = predictions.index.intersection(predictions_returns_only.index)
        if common_idx.empty:
            print("No common time periods found between prediction sets for t-test.")
            return

        pred_full_common_idx = predictions.loc[common_idx]
        pred_only_common_idx = predictions_returns_only.loc[common_idx]

        common_cols = pred_full_common_idx.columns.intersection(pred_only_common_idx.columns)
        if common_cols.empty:
            print("No common stocks found between prediction sets for t-test.")
            return
        
        pred_full_aligned = pred_full_common_idx[common_cols]
        pred_only_aligned = pred_only_common_idx[common_cols]

        mean_preds_full = pred_full_aligned.mean(axis=1, skipna=True)
        mean_preds_only = pred_only_aligned.mean(axis=1, skipna=True)
        
        paired_data = pd.DataFrame({'full': mean_preds_full, 'only': mean_preds_only}).dropna()

        if len(paired_data) < 2:
            print(f"Not enough overlapping data points ({len(paired_data)}) for t-test after aggregation and NaN removal.")
            return
        
        try:
            t_statistic, p_value = stats.ttest_rel(paired_data['full'], paired_data['only'])
        except ValueError as e:
            print(f"Error during t-test calculation (e.g., zero variance in differences): {e}")
            return

        print(f"--- Paired T-test: Full Strategy vs. Returns-Only Strategy (based on mean predictions across stocks) ---")
        if np.isnan(t_statistic) or np.isnan(p_value):
            print(f"Number of paired observations used in t-test: {len(paired_data)}")
            print("T-test resulted in NaN. This often occurs if the two series of mean predictions are identical or have no variance in their differences.")
            if np.array_equal(paired_data['full'].values, paired_data['only'].values):
                 print("The mean predictions for both strategies are identical over the common period.")
        else:
            print(f"Number of paired observations used in t-test: {len(paired_data)}")
            print(f"T-statistic: {t_statistic:.4f}")
            print(f"P-value: {p_value:.4f}")

            alpha_significance = 0.05
            if p_value < alpha_significance:
                print(f"The difference in mean aggregated monthly predictions is statistically significant at the {alpha_significance*100:.0f}% level.")
                if t_statistic > 0: # ttest_rel(a,b) tests mean(a-b). 'a' is 'full', 'b' is 'only'.
                    print("The 'full' strategy has significantly higher mean aggregated predictions.")
                else:
                    print("The 'returns_only' strategy has significantly higher mean aggregated predictions.")
            else:
                print(f"There is no statistically significant difference in mean aggregated monthly predictions at the {alpha_significance*100:.0f}% level.")
      
    def jensens_alpha(self, returns_dict, risk_free_csv="risk-free-rate.csv"):
        print("\n=== Jensen's Alpha Analysis ===")
        
        from scipy.stats import linregress
        import pandas as pd
        

        rf_data = pd.read_csv(risk_free_csv, sep=';')
        rf_data['Month'] = pd.to_datetime(rf_data['Month'], format='%Y-%m')
        rf_data = rf_data.set_index('Month').sort_index()

        rf_data['Monthly_Rate'] = rf_data['Policy Rate'] / 100
        
        index_returns = np.array(returns_dict["index_returns"])
        
        if index_returns.ndim > 1:
            index_returns = index_returns[:, 0]

        n_periods = len(index_returns)
        
        risk_free_rates = rf_data['Monthly_Rate'].tail(n_periods).values
        
        market_excess_returns = index_returns - risk_free_rates
        
        strategies = ["full_returns", "returns_only_returns"]
        
        for strategy in strategies:
            strategy_returns = np.array(returns_dict[strategy])
            
            if strategy_returns.ndim > 1:
                strategy_returns = strategy_returns[:, 0]
                
            strategy_excess_returns = strategy_returns - risk_free_rates
            
            slope, intercept, r_value, p_value, std_err = linregress(
                market_excess_returns, strategy_excess_returns
            )
            
            jensens_alpha_value = intercept
            beta = slope
            
            annualized_alpha = jensens_alpha_value * 12
            
            print(f"\n--- {strategy.replace('_', ' ').title()} ---")
            print(f"  Jensen's Alpha (monthly): {jensens_alpha_value:.6f}")
            print(f"  Jensen's Alpha (annualized): {annualized_alpha:.4f}")
            print(f"  Beta: {beta:.4f}")
            print(f"  R-squared: {r_value**2:.4f}")
            print(f"  P-value for alpha: {p_value:.6f}")
            print(f"  Standard Error: {std_err:.6f}")
            print(f"  Average Risk-Free Rate Used: {np.mean(risk_free_rates)*100:.3f}% monthly")
            
            if p_value < 0.05:
                if jensens_alpha_value > 0:
                    print(f"  → Strategy significantly outperforms market (positive alpha)")
                else:
                    print(f"  → Strategy significantly underperforms market (negative alpha)")
            else:
                print(f"  → No significant outperformance/underperformance vs market")
            
            if beta > 1:
                print(f"  → Higher risk than market (beta > 1)")
            elif beta < 1:
                print(f"  → Lower risk than market (beta < 1)")
            else:
                print(f"  → Similar risk to market (beta ≈ 1)")
        
        full_returns = np.array(returns_dict["full_returns"])
        returns_only_returns = np.array(returns_dict["returns_only_returns"])
        
        if full_returns.ndim > 1:
            full_returns = full_returns[:, 0]
        if returns_only_returns.ndim > 1:
            returns_only_returns = returns_only_returns[:, 0]
        
        full_excess = full_returns - risk_free_rates
        returns_only_excess = returns_only_returns - risk_free_rates
        
        _, full_alpha, _, _, _ = linregress(market_excess_returns, full_excess)
        _, returns_only_alpha, _, _, _ = linregress(market_excess_returns, returns_only_excess)
        
        print(f"\n--- Alpha Comparison ---")
        print(f"  Full Strategy Alpha: {full_alpha:.6f}")
        print(f"  Returns-Only Strategy Alpha: {returns_only_alpha:.6f}")
        print(f"  Difference: {full_alpha - returns_only_alpha:.6f}")
        
        if full_alpha > returns_only_alpha:
            print(f"  → Full Strategy has better risk-adjusted performance")
        else:
            print(f"  → Returns-Only Strategy has better risk-adjusted performance")