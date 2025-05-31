# stock_modeler_package/model_tuner.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV

class ModelTuner:
    def __init__(self, data_handler, tscv_tuning_splits=5, random_search_n_iter=50):
        self.data_handler = data_handler
        self.tscv_tuning = TimeSeriesSplit(n_splits=tscv_tuning_splits)
        self.random_search_n_iter = random_search_n_iter

        self.param_grid_random = {
            'n_estimators': [20, 50, 100, 200, 400, 600, 1000],
            'max_depth': [None, 5, 10, 15, 20, 30, 100, 150, 500],
            'min_samples_leaf': [1, 2, 4, 6, 8, 15, 50],
            'min_samples_split': [2, 5, 10, 15, 20, 50],
            'max_features': ['sqrt', 'log2', None]
        }
        
        """self.param_grid_random = {
            'n_estimators': [20],
            'max_depth': [5],
            'min_samples_leaf': [1],
            'min_samples_split': [2],
            'max_features': ['sqrt']
        }"""

    def _generate_grid_search_params(self, best_params_random, random_param_grid):
        max_depth_values = []
        best_md_from_random = best_params_random['max_depth']
        if best_md_from_random is None:
            max_depth_values.extend([20, 30, 50, None])
        else:
            step = 10 if best_md_from_random > 30 else 5
            max_depth_values.extend([
                max(5, best_md_from_random - step),
                best_md_from_random,
                min(150, best_md_from_random + step)
            ])
            if best_md_from_random >= 30:
                max_depth_values.append(None)

        _unique_md_values = list(set(max_depth_values))
        _numeric_md_sorted = sorted([d for d in _unique_md_values if d is not None and d >= 5])
        _final_md_grid_options = _numeric_md_sorted
        if None in _unique_md_values:
            _final_md_grid_options.append(None)

        best_leaf_val = best_params_random['min_samples_leaf']
        leaf_options_builder = [
            max(1, best_leaf_val - 2), max(1, best_leaf_val - 1),
            best_leaf_val,
            best_leaf_val + 1, best_leaf_val + 2
        ]
        max_leaf_from_random_grid = max(random_param_grid['min_samples_leaf'])
        final_leaf_options = sorted(list(set(opt for opt in leaf_options_builder if 1 <= opt <= max_leaf_from_random_grid)))
        if not final_leaf_options:
            final_leaf_options = [1] if best_leaf_val < 1 else [min(best_leaf_val, max_leaf_from_random_grid)]

        best_split_val = best_params_random['min_samples_split']
        split_options_builder = [
            max(2, best_split_val - 4), max(2, best_split_val - 2),
            best_split_val,
            best_split_val + 2, best_split_val + 4
        ]
        max_split_from_random_grid = max(random_param_grid['min_samples_split'])
        final_split_options = sorted(list(set(opt for opt in split_options_builder if 2 <= opt <= max_split_from_random_grid)))
        if not final_split_options:
            final_split_options = [2] if best_split_val < 2 else [min(best_split_val, max_split_from_random_grid)]

        param_grid_grid = {
            'n_estimators': sorted(list(set([
                max(50, best_params_random['n_estimators'] - 100),
                best_params_random['n_estimators'],
                min(1000, best_params_random['n_estimators'] + 100)
            ]))),
            'max_depth': _final_md_grid_options,
            'min_samples_leaf': final_leaf_options,
            'min_samples_split': final_split_options,
            'max_features': [best_params_random['max_features']]
        }
        return param_grid_grid

    def tune_models(self, model_type):
        full_target_names = self.data_handler.full_target_names
        
        current_run_params = {}

        for full_target_name_from_Y in full_target_names:
 
            
            X_for_current_model = self.data_handler.getFeaturesFromStock(full_target_name_from_Y, model_type=model_type)
            current_Y_target = self.data_handler.Y_all_targets[full_target_name_from_Y]

            rf_model = RandomForestRegressor(random_state=42, bootstrap=True)
            
            print(f"Starting RandomizedSearchCV for {full_target_name_from_Y}, with modeltype: {model_type}...")
            random_search = RandomizedSearchCV(
                estimator=rf_model,
                param_distributions=self.param_grid_random,
                n_iter=self.random_search_n_iter, 
                cv=self.tscv_tuning, 
                scoring='r2',
                random_state=42,
                n_jobs=2,
                verbose=1
            )
            random_search.fit(X_for_current_model, current_Y_target)
            print(f"Best params from RandomizedSearch for {full_target_name_from_Y}, with modeltype: {model_type}: {random_search.best_params_}")
            param_grid_grid = self._generate_grid_search_params(random_search.best_params_, self.param_grid_random)
            
            print(f"\nRefined Grid for GridSearchCV ({full_target_name_from_Y}), with modeltype: {model_type}:")
            for param, values in param_grid_grid.items(): print(f"{param}: {values}")
            
            print(f"\nStarting GridSearchCV for {full_target_name_from_Y}, with modeltype: {model_type}...")
            grid_search = GridSearchCV(
                estimator=rf_model, 
                param_grid=param_grid_grid,
                cv=self.tscv_tuning,
                scoring='r2',
                n_jobs=2,
                verbose=1
            )
            grid_search.fit(X_for_current_model, current_Y_target)
            
            current_run_params[full_target_name_from_Y] = grid_search.best_params_
            print(f"\nFinal RF parameters for {full_target_name_from_Y}, with modeltype: {model_type}: {grid_search.best_params_}")
            
        return current_run_params