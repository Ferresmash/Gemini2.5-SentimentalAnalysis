# main_runner.py
import os
from stock_modeler_package.pipeline import StockModelerPipeline

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline = StockModelerPipeline(current_dir) 
    pipeline.run()