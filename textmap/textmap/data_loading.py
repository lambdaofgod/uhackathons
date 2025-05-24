import pandas as pd
import tiktoken
import logging

# Get logger
logger = logging.getLogger(__name__)


def _validate_inputs(file_input, text_column, title_column, date_column):
    """
    Validates the input parameters.

    Args:
        file_input: The uploaded file object from Gradio.
        text_column (str): The column containing the text data.
        title_column (str): The column containing the title data.
        date_column (str): The column containing the date data.

    Returns:
        tuple: A tuple containing:
            - error_message (str or None): Error message if validation fails, None otherwise.
    """
    if file_input is None:
        return "Please upload a file and select time granularity."

    # Only text and title columns are required
    if not text_column or not title_column:
        return "Please select columns for text and title."

    # date_column can be None or "None"
    return None


def _load_dataframe(file_input):
    """
    Loads a dataframe from a file.

    Args:
        file_input: The uploaded file object from Gradio.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame or None): The loaded DataFrame, or None if an error occurred.
            - error_message (str or None): Error message if loading fails, None otherwise.
    """
    try:
        file_ext = file_input.name.split(".")[-1].lower()

        if file_ext == "jsonl":
            df = pd.read_json(file_input.name, lines=True)
        elif file_ext == "csv":
            df = pd.read_csv(file_input.name)
        else:
            return (
                None,
                f"Unsupported file format: {file_ext}. Please upload a CSV or JSONL file.",
            )

        return df, None
    except Exception as e:
        return None, f"Error reading file: {e}"


def _check_required_columns(df, required_columns):
    """
    Checks if the dataframe contains all required columns.

    Args:
        df (pd.DataFrame): The dataframe to check.
        required_columns (list): List of required column names.

    Returns:
        tuple: A tuple containing:
            - is_valid (bool): True if all required columns are present, False otherwise.
            - error_message (str or None): Error message if validation fails, None otherwise.
    """
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        return False, f"Selected columns not found in file: {', '.join(missing_cols)}"

    return True, None


def _standardize_dataframe(df, text_column, title_column, date_column):
    """
    Standardizes the dataframe by renaming columns and converting date.

    Args:
        df (pd.DataFrame): The dataframe to standardize.
        text_column (str): The column containing the text data.
        title_column (str): The column containing the title data.
        date_column (str): The column containing the date data.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame or None): The standardized DataFrame, or None if an error occurred.
            - error_message (str or None): Error message if standardization fails, None otherwise.
    """
    try:
        # Create a standardized DataFrame with the expected column names
        data = {
            "text": df[text_column],
            "title": df[title_column],
        }
        
        # Only add date if date_column is provided
        if date_column is not None:
            try:
                logger.info(f"Converting date column: {date_column}")
                data["date"] = pd.to_datetime(df[date_column])
                logger.info(f"Date conversion successful")
            except Exception as e:
                error_msg = f"Error parsing 'date' column: {e}. Ensure dates are in a recognizable format."
                logger.info(error_msg)
                return None, error_msg
        
        standardized_df = pd.DataFrame(data)
        return standardized_df, None
    except Exception as e:
        error_msg = f"Error standardizing dataframe: {e}"
        logger.info(error_msg)
        return None, error_msg


def _filter_by_token_count(df, min_tokens):
    """
    Filters the dataframe to keep only rows with sufficient tokens.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        min_tokens (int): Minimum number of tokens required for a text to be included.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The filtered DataFrame.
            - filtered_count (int): Number of rows that were filtered out.
    """
    if min_tokens <= 0:
        return df, 0

    # Get the cl100k_base encoder which is used by many models including GPT-4
    enc = tiktoken.get_encoding("cl100k_base")

    # Count tokens for each text
    disallowed_special = enc.special_tokens_set - {"<|endoftext|>"}
    token_counts = df["text"].apply(
        lambda x: len(enc.encode(x, disallowed_special=disallowed_special))
    )

    # Filter the DataFrame to keep only rows with sufficient tokens
    original_count = len(df)
    filtered_df = df[token_counts >= min_tokens].reset_index(drop=True)
    filtered_count = original_count - len(filtered_df)

    if filtered_count > 0:
        print(
            f"Filtered out {filtered_count} texts with fewer than {min_tokens} tokens."
        )

    return filtered_df, filtered_count


def _assign_time_periods(df, time_granularity):
    """
    Assigns time periods to the dataframe based on the specified granularity.

    Args:
        df (pd.DataFrame): The dataframe to process.
        time_granularity (str): The selected time granularity ('day', 'week', 'month').

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame or None): The processed DataFrame, or None if an error occurred.
            - error_message (str or None): Error message if processing fails, None otherwise.
    """
    # Sort by date before assigning periods to ensure consistent period calculation
    df = df.sort_values(by="date").reset_index(drop=True)

    if time_granularity == "day":
        # Using ordinal day number since epoch for a unique integer ID per day
        df["time_period"] = df["date"].dt.to_period("D").apply(lambda p: p.ordinal)
    elif time_granularity == "week":
        # Using ordinal week number since epoch for a unique integer ID per week
        df["time_period"] = df["date"].dt.to_period("W").apply(lambda p: p.ordinal)
    elif time_granularity == "month":
        # Using ordinal month number since epoch for a unique integer ID per month
        df["time_period"] = df["date"].dt.to_period("M").apply(lambda p: p.ordinal)
    else:
        return None, "Invalid time granularity selected."

    return df, None


def load_and_preprocess_data(
    file_input, time_granularity, text_column, title_column, date_column, min_tokens=64
):
    """
    Loads data from a JSONL or CSV file, preprocesses it, and calculates the time_period.

    Args:
        file_input: The uploaded file object from Gradio.
        time_granularity (str): The selected time granularity ('day', 'week', 'month').
        text_column (str): The column containing the text data.
        title_column (str): The column containing the title data.
        date_column (str): The column containing the date data.
        min_tokens (int): Minimum number of tokens required for a text to be included. Default is 64.

    Returns:
        tuple: A tuple containing:
            - status_message (str): A message indicating the outcome.
            - df (pd.DataFrame or None): The processed DataFrame, or None if an error occurred.
    """
    logger.info(f"Starting data preprocessing: file={file_input.name}, columns={text_column},{title_column},{date_column}")
    
    # Handle the case where date_column is "None"
    actual_date_column = None if date_column == "None" else date_column
    logger.info(f"Using actual_date_column: {actual_date_column}")
    
    # Validate inputs
    error_message = _validate_inputs(file_input, text_column, title_column, actual_date_column)
    if error_message:
        logger.info(f"Input validation failed: {error_message}")
        return error_message, None

    # Load dataframe
    logger.info(f"Loading dataframe from {file_input.name}")
    df, error_message = _load_dataframe(file_input)
    if error_message:
        logger.info(f"Failed to load dataframe: {error_message}")
        return error_message, None
    logger.info(f"Dataframe loaded with shape: {df.shape}")

    # Check required columns
    required_columns = [text_column, title_column]
    if actual_date_column is not None:
        required_columns.append(actual_date_column)
    
    logger.info(f"Checking for required columns: {required_columns}")
    is_valid, error_message = _check_required_columns(df, required_columns)
    if not is_valid:
        logger.info(f"Required columns check failed: {error_message}")
        return error_message, None

    # Standardize dataframe
    logger.info("Standardizing dataframe")
    df, error_message = _standardize_dataframe(
        df, text_column, title_column, actual_date_column
    )
    if error_message:
        logger.info(f"Failed to standardize dataframe: {error_message}")
        return error_message, None
    logger.info(f"Dataframe standardized with shape: {df.shape}")

    # Filter by token count
    logger.info(f"Filtering by token count (min_tokens={min_tokens})")
    df, filtered_count = _filter_by_token_count(df, min_tokens)
    logger.info(f"After filtering: {len(df)} rows remain, {filtered_count} filtered out")

    # Assign time periods
    if actual_date_column is not None and "date" in df.columns:
        logger.info(f"Assigning time periods with granularity: {time_granularity}")
        df, error_message = _assign_time_periods(df, time_granularity)
        if error_message:
            logger.info(f"Failed to assign time periods: {error_message}")
            return error_message, None
        logger.info("Time periods assigned successfully")
    else:
        logger.info("Skipping time period assignment (no date column)")

    logger.info(f"Data preprocessing complete: {len(df)} records ready for analysis")
    return (
        f"File '{file_input.name}' processed. {len(df)} records loaded. Time granularity: {time_granularity}. Minimum token count: {min_tokens}.",
        df,
    )
