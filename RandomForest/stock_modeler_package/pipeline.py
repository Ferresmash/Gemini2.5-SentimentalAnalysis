# stock_modeler_package/pipeline.py
from .data_handler import DataHandler
from .model_tuner import ModelTuner
from .model_evaluator import ModelEvaluator
from .portfolio_simulator import PortfolioSimulator
from .result_visualizer import ResultVisualizer


class StockModelerPipeline:

    def __init__(self, filepath, random_search_n_iter=50, tscv_tuning_splits=3, tscv_eval_splits=5):
        self.filepath = filepath
        self.random_search_n_iter = random_search_n_iter
        self.tscv_tuning_splits = tscv_tuning_splits
        self.tscv_eval_splits = tscv_eval_splits
        
        self.data_handler = None
        self.model_tuner = None
        self.model_evaluator = None
        self.result_visualizer = None

    def run(self):
        
        print("--- 1. Loading and Preparing Data ---")
        try:
            self.data_handler = DataHandler(self.filepath)
            load_success = self.data_handler.load_and_prepare_data()

            if not load_success:
                print("Data loading failed. Please check the file path and format.")
                return
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return        

        print("\n--- 2. Tuning Models ---")
        self.model_tuner = ModelTuner(
            self.data_handler, 
            tscv_tuning_splits=self.tscv_tuning_splits,
            random_search_n_iter=self.random_search_n_iter
        )
        tuned_params = self.model_tuner.tune_models(model_type="full")
        tuned_params_returns_only = self.model_tuner.tune_models(model_type="returns_only")

        print("\n--- 3. Evaluating Models and Extracting Results ---")
        self.model_evaluator = ModelEvaluator(
            self.data_handler, 
            tuned_params,
            tuned_params_returns_only,
            tscv_eval_splits=self.tscv_eval_splits
        )
        
        full_predictions = self.model_evaluator.evaluate_models(model_type="full")
        only_returns_predictions = self.model_evaluator.evaluate_models(model_type="returns_only")
        self.model_evaluator.t_test(full_predictions,only_returns_predictions)
        
        print("\n--- 4. Simulating portfolios ---")
        
        actual_returns = self.data_handler.Y_all_targets.copy()
        
        full_strategy_sim = PortfolioSimulator(
            full_predictions,
            actual_returns
        )
        only_returns_strategy_sim = PortfolioSimulator(
            only_returns_predictions, 
            actual_returns
        )
        
        initial_capital = 1000.0
        top_k = 6
        bottom_k = 6
        
        all_features_results = full_strategy_sim.simulate_strategy(top_k=top_k, bottom_k=bottom_k, initial_capital=initial_capital)
        returns_only_results = only_returns_strategy_sim.simulate_strategy(top_k=top_k, bottom_k=bottom_k, initial_capital=initial_capital)
        index_results = full_strategy_sim.simulate_index(initial_capital=initial_capital)

        returns_dict = {
            "full_returns": all_features_results,
            "returns_only_returns": returns_only_results,
            "index_returns": index_results
        }
        
        print("\n--- 5. Evaluation on portfolio performance  ---")
        
        self.model_evaluator.jensens_alpha(returns_dict)
        self.model_evaluator.t_test_portfolio(returns_dict)
        
        
        print("\n--- 6. Visualizing Results ---")
        self.result_visualizer = ResultVisualizer( self.data_handler, returns_dict)
        self.result_visualizer.show_all()
        
        print("\n--- Pipeline Execution Complete ---")