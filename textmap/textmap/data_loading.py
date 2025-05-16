import pandas as pd

def load_and_preprocess_data(file_input, time_granularity, text_column, title_column, date_column):
    """
    Loads data from a JSONL or CSV file, preprocesses it, and calculates the time_period.

    Args:
        file_input: The uploaded file object from Gradio.
        time_granularity (str): The selected time granularity ('day', 'week', 'month').
        text_column (str): The column containing the text data.
        title_column (str): The column containing the title data.
        date_column (str): The column containing the date data.

    Returns:
        tuple: A tuple containing:
            - status_message (str): A message indicating the outcome.
            - df (pd.DataFrame or None): The processed DataFrame, or None if an error occurred.
    """
    if file_input is None:
        return "Please upload a file and select time granularity.", None
    
    if not all([text_column, title_column, date_column]):
        return "Please select columns for text, title, and date.", None

    try:
        file_ext = file_input.name.split('.')[-1].lower()
        
        if file_ext == 'jsonl':
            df = pd.read_json(file_input.name, lines=True)
        elif file_ext == 'csv':
            df = pd.read_csv(file_input.name)
        else:
            return f"Unsupported file format: {file_ext}. Please upload a CSV or JSONL file.", None
    except Exception as e:
        return f"Error reading file: {e}", None

    required_columns = [text_column, title_column, date_column]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        return f"Selected columns not found in file: {', '.join(missing_cols)}", None

    try:
        # Create a standardized DataFrame with the expected column names
        standardized_df = pd.DataFrame({
            'text': df[text_column],
            'title': df[title_column],
            'date': pd.to_datetime(df[date_column])
        })
        
        # Replace the original DataFrame with the standardized one
        df = standardized_df
    except Exception as e:
        return f"Error parsing 'date' column: {e}. Ensure dates are in a recognizable format.", None

    # Sort by date before assigning periods to ensure consistent period calculation if needed later
    df = df.sort_values(by='date').reset_index(drop=True)

    if time_granularity == "day":
        # Using ordinal day number since epoch for a unique integer ID per day
        df['time_period'] = df['date'].dt.to_period('D').apply(lambda p: p.ordinal)
    elif time_granularity == "week":
        # Using ordinal week number since epoch for a unique integer ID per week
        # .dt.isocalendar().week could also be used if week numbers should reset each year,
        # but for continuous time periods, ordinal is better.
        df['time_period'] = df['date'].dt.to_period('W').apply(lambda p: p.ordinal)
    elif time_granularity == "month":
        # Using ordinal month number since epoch for a unique integer ID per month
        df['time_period'] = df['date'].dt.to_period('M').apply(lambda p: p.ordinal)
    else:
        return "Invalid time granularity selected.", None
    
    return f"File '{file_input.name}' processed. {len(df)} records loaded. Time granularity: {time_granularity}.", df
