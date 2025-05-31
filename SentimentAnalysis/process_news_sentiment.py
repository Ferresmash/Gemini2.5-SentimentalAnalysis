import os
import csv
import google.generativeai as genai
from pathlib import Path
import time
import re

# --- Configuration ---
API_KEY_FILE = "apiKey.txt"
PRE_PROMPT_FILE = "PrePrompt.txt"
NEWS_BASE_FOLDER = "news_files(21-23)/"
API_CALL_DELAY_SECONDS = 1
GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-05-06"

STOCK_TICKERS = [
    'NDA-SE'
]
"""
STOCK_TICKERS = [
    'ABB', 'ALFA', 'ASSA-B', 'AZN', 'ATCO-A',
    'BOL', 'ELUX', 'ERIC-B', 'ESSITY-B', 'EVO', 'GETI-B',
    'SHB-A', 'HM-B', 'HEXA-B', 'INVE-B', 'KINV-B', 'NIBE-B',
    'NDA-SE', 'SAAB-B', 'SBB-B', 'SAND', 'SCA-B', 'SEB-A',
    'SINCH', 'SKF-B', 'SWED-A', 'TEL2-B', 'TELIA', 'VOLV-B'
]
"""


def load_api_key(filepath=API_KEY_FILE):
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file '{filepath}' not found.")
        return None

def load_pre_prompt_template(filepath=PRE_PROMPT_FILE):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Pre-prompt file '{filepath}' not found.")
        return None

def get_sentiment_from_gemini(api_key, model_name, formatted_pre_prompt, news_text):
    if not api_key:
        print("API key not configured. Skipping Gemini call.")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    if not formatted_pre_prompt.endswith('\n'):
        formatted_pre_prompt += '\n'
    full_prompt = f"{formatted_pre_prompt}{news_text}"

    try:
        if len(full_prompt) > 30000 * 4 and model_name == "gemini-pro":
             print(f"Warning: Prompt for a file might be too long ({len(full_prompt)} chars). Truncating news.")
             news_text_truncated = news_text[:100000]
             full_prompt = f"{formatted_pre_prompt}{news_text_truncated}"

        response = model.generate_content(full_prompt)
        sentiment_score_str = response.text.strip()
        return float(sentiment_score_str)
    except ValueError:
        print(f"Error: Gemini did not return a valid number. Response: '{sentiment_score_str}'")
        return None
    except Exception as e:
        print(f"An error occurred while calling Gemini API: {e}")
        if hasattr(response, 'prompt_feedback'):
            print(f"Prompt Feedback: {response.prompt_feedback}")
        return None

def process_ticker(ticker, api_key, pre_prompt_template):
    print(f"\n\n{'='*50}")
    print(f"--- Starting sentiment analysis for stock: {ticker} ---")
    print(f"{'='*50}")
    
    news_folder = os.path.join(NEWS_BASE_FOLDER, ticker)
    news_folder_path = Path(news_folder)
    
    if not news_folder_path.is_dir():
        print(f"Error: News folder '{news_folder}' not found. Skipping {ticker}.")
        return

    results = []
    txt_files = sorted([f for f in news_folder_path.iterdir() if f.is_file() and f.suffix.lower() == '.txt' and re.match(r"^\d{4}-\d{2}$", f.stem)])
    
    if not txt_files:
        print(f"No 'yyyy-mm.txt' files found in '{news_folder}'. Ensure filenames match this format (e.g., 2023-01.txt).")
        return

    print(f"Found {len(txt_files)} 'yyyy-mm.txt' files to process.")

    for txt_file_path in txt_files:
        month_year_identifier = txt_file_path.stem
        print(f"\nProcessing: {txt_file_path.name}...")

        try:
            current_formatted_pre_prompt = pre_prompt_template.format(
                STOCK_TICKER=ticker,
                MONTH_YEAR=month_year_identifier
            )
        except KeyError as e:
            print(f"Error: Placeholder {e} not found or mismatch in PrePrompt.txt. Make sure it contains {{STOCK_TICKER}} and {{MONTH_YEAR}}.")
            results.append((month_year_identifier, "Prompt Formatting Error"))
            continue

        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                news_content = f.read()
        except Exception as e:
            print(f"Error reading file {txt_file_path.name}: {e}")
            results.append((month_year_identifier, "Error Reading File"))
            continue

        if not news_content.strip():
            print(f"File {txt_file_path.name} is empty. Skipping.")
            results.append((month_year_identifier, "Empty File"))
            continue

        sentiment_score = get_sentiment_from_gemini(api_key, GEMINI_MODEL_NAME, current_formatted_pre_prompt, news_content)

        if sentiment_score is not None:
            print(f"Sentiment score for {month_year_identifier}: {sentiment_score}")
            results.append((month_year_identifier, sentiment_score))
        else:
            print(f"Failed to get sentiment score for {month_year_identifier}.")
            results.append((month_year_identifier, "Error in API"))
        
        if API_CALL_DELAY_SECONDS > 0 and txt_file_path != txt_files[-1]:
            print(f"Waiting for {API_CALL_DELAY_SECONDS}s...")
            time.sleep(API_CALL_DELAY_SECONDS)

    output_csv_filename = f"{ticker}_sentiment_scores.csv"
    try:
        with open(output_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Month_Year", "Sentiment_Score"])
            for row in results:
                writer.writerow(row)
        print(f"\nSentiment scores successfully saved to '{output_csv_filename}'")
    except IOError:
        print(f"Error: Could not write to CSV file '{output_csv_filename}'.")
    
    return results

def main():
    api_key = load_api_key()
    pre_prompt_template = load_pre_prompt_template()

    if not api_key or not pre_prompt_template:
        print("Exiting due to missing API key or pre-prompt template.")
        return


    all_results = {}
    for ticker in STOCK_TICKERS:
        ticker_results = process_ticker(ticker, api_key, pre_prompt_template)
        if ticker_results:
            all_results[ticker] = ticker_results
            
        if ticker != STOCK_TICKERS[-1]:
            delay = API_CALL_DELAY_SECONDS * 2 
            print(f"\nMoving to next ticker. Waiting for {delay}s...")
            time.sleep(delay)
    
    print("\nCompleted sentiment analysis for all tickers!")

if __name__ == "__main__":
    main()