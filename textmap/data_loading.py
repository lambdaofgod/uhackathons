import pandas as pd

def load_and_preprocess_data(jsonl_file, time_granularity):
    """
    Loads data from a JSONL file, preprocesses it, and calculates the time_period.

    Args:
        jsonl_file: The uploaded file object from Gradio.
        time_granularity (str): The selected time granularity ('day', 'week', 'month').

    Returns:
        tuple: A tuple containing:
            - status_message (str): A message indicating the outcome.
            - df (pd.DataFrame or None): The processed DataFrame, or None if an error occurred.
    """
    if jsonl_file is None:
        return "Please upload a file and select time granularity.", None

    try:
        # Use jsonl_file.name to get the path of the temporary uploaded file
        df = pd.read_json(jsonl_file.name, lines=True)
    except Exception as e:
        return f"Error reading JSONL file: {e}", None

    required_columns = ["text", "title", "date"]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        return f"JSONL file must contain the columns: {', '.join(required_columns)}. Missing: {', '.join(missing_cols)}", None

    try:
        df['date'] = pd.to_datetime(df['date'])
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
    
    return f"File '{jsonl_file.name}' processed. {len(df)} records loaded. Time granularity: {time_granularity}.", df
