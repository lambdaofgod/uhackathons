import gradio as gr
import pandas as pd
import json
import logging

# Set up logging
from textmap.logging_setup import setup_logging
logger = setup_logging()

# Assuming app.py and data_loading.py are in the same directory 'textmap'
# and you run the app from the parent directory of 'textmap' (e.g., python -m textmap.app)
# or from within 'textmap' (e.g., python app.py)
from textmap.data_loading import load_and_preprocess_data
from textmap.dynamic_topic_models import DynamicTopicModel
from textmap.bertopic_utils import BERTopicUtils


def load_file_preview(file_path):
    """
    Load a preview of the file and return column names and a sample

    Args:
        file_path: The uploaded file object from Gradio

    Returns:
        tuple: Lists of column names, preview dataframe, file format, and status message
    """
    if file_path is None:
        return [], None, None, "Please upload a file first."

    file_ext = file_path.name.split(".")[-1].lower()

    try:
        if file_ext == "jsonl":
            # Read a few lines for preview
            with open(file_path.name, "r") as f:
                lines = [f.readline().strip() for _ in range(5)]
                data = [json.loads(line) for line in lines if line]

            if not data:
                return [], None, None, "No valid data found in the JSONL file."

            # Create a DataFrame from the sample
            preview_df = pd.DataFrame(data)
            columns = list(preview_df.columns)
            return (
                columns,
                preview_df.head(5),
                "jsonl",
                f"Successfully loaded JSONL preview with {len(columns)} columns.",
            )

        elif file_ext == "csv":
            preview_df = pd.read_csv(file_path.name, nrows=5)
            columns = list(preview_df.columns)
            return (
                columns,
                preview_df,
                "csv",
                f"Successfully loaded CSV preview with {len(columns)} columns.",
            )

        return (
            [],
            None,
            None,
            f"Unsupported file format: {file_ext}. Please upload a CSV or JSONL file.",
        )
    except Exception as e:
        return [], None, None, f"Error loading file: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Topic Modeling Visualization")

    # Gradio State to store the file path, format, and model
    file_info = gr.State(None)
    df_state = gr.State(None)
    model_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):  # Left sidebar
            gr.Markdown("## Inputs")

            # Step 1: File Upload
            with gr.Group():
                gr.Markdown("### Step 1: Upload Data")
                file_input = gr.File(
                    label="Upload Data File",
                    file_types=[".jsonl", ".csv"],
                )
                preview_button = gr.Button("Load File Preview")
                preview_status = gr.Textbox(label="Preview Status", interactive=False)

            # Step 2: Column Selection (initially hidden)
            with gr.Group(visible=False) as column_selection_group:
                gr.Markdown("### Step 2: Select Columns")
                preview_output = gr.DataFrame(label="Data Preview")

                text_column = gr.Dropdown(
                    label="Text Column", choices=[], interactive=True
                )
                title_column = gr.Dropdown(
                    label="Title Column", choices=[], interactive=True
                )
                date_column = gr.Dropdown(
                    label="Date Column", 
                    choices=[], 
                    value="None",
                    interactive=True,
                    info="Select 'None' to skip time-based visualization"
                )

                granularity_input = gr.Radio(
                    label="Time Granularity",
                    choices=["day", "week", "month"],
                    value="month",
                )

                min_tokens_input = gr.Number(
                    label="Minimum Token Count",
                    value=15,
                    precision=0,
                    info="Minimum number of tokens required for a text to be included in the analysis",
                )

                representation_model = gr.Dropdown(
                    label="Representation Model",
                    choices=[
                        "Default",
                        "KeyBERTInspired",
                        "OpenAI",
                        "MaximalMarginalRelevance",
                    ],
                    value="Default",
                    info="Select the representation model for BERTopic\nFor information on representation models see\nhttps://maartengr.github.io/BERTopic/getting_started/representation/representation.html",
                )

                submit_button = gr.Button("Visualize Topics")

        with gr.Column(scale=3):  # Main content area
            gr.Markdown("## Topic Visualization")
            output_display = gr.Textbox(
                label="Status", interactive=False
            )  # Placeholder for status messages

            # Add a table to display topics
            topics_table = gr.DataFrame(
                label="Topics", interactive=False, visible=False
            )

            # Add a plotly figure for topics over time visualization
            topics_over_time_plot = gr.Plot(label="Topics Over Time", visible=False)

    # Connect preview button to load file preview
    def load_file_preview_and_update_ui(file_path):
        """
        Load a preview of the file and update the UI accordingly

        Args:
            file_path: The uploaded file object from Gradio

        Returns:
            dict: Updates for the UI components
        """
        if file_path is None:
            return {
                text_column: gr.update(choices=[], value=None),
                title_column: gr.update(choices=[], value=None),
                date_column: gr.update(choices=[], value=None),
                preview_output: gr.update(value=None),
                column_selection_group: gr.update(visible=False),
                file_info: None,
                preview_status: "Please upload a file first.",
            }

        file_ext = file_path.name.split(".")[-1].lower()

        try:
            if file_ext == "jsonl":
                # Read a few lines for preview
                with open(file_path.name, "r") as f:
                    lines = [f.readline().strip() for _ in range(5)]
                    data = [json.loads(line) for line in lines if line]

                if not data:
                    return {
                        text_column: gr.update(choices=[], value=None),
                        title_column: gr.update(choices=[], value=None),
                        date_column: gr.update(choices=[], value=None),
                        preview_output: gr.update(value=None),
                        column_selection_group: gr.update(visible=False),
                        file_info: None,
                        preview_status: "No valid data found in the JSONL file.",
                    }

                # Create a DataFrame from the sample
                preview_df = pd.DataFrame(data)
                columns = list(preview_df.columns)

            elif file_ext == "csv":
                preview_df = pd.read_csv(file_path.name, nrows=5)
                columns = list(preview_df.columns)

            else:
                return {
                    text_column: gr.update(choices=[], value=None),
                    title_column: gr.update(choices=[], value=None),
                    date_column: gr.update(choices=[], value=None),
                    preview_output: gr.update(value=None),
                    column_selection_group: gr.update(visible=False),
                    file_info: None,
                    preview_status: f"Unsupported file format: {file_ext}. Please upload a CSV or JSONL file.",
                }

            # Make intelligent guesses for column selection
            default_text = columns[0] if columns else None
            default_title = columns[0] if columns else None
            default_date = columns[0] if columns else None

            # Try to make intelligent guesses for column selection
            if columns:
                # Look for common text column names
                text_candidates = ["text", "body", "content", "description"]
                for candidate in text_candidates:
                    if candidate in columns:
                        default_text = candidate
                        break

                # Look for common title column names
                title_candidates = ["title", "heading", "name", "subject"]
                for candidate in title_candidates:
                    if candidate in columns:
                        default_title = candidate
                        break

                # Look for common date column names
                date_candidates = ["date", "created_at", "timestamp", "time"]
                for candidate in date_candidates:
                    if candidate in columns:
                        default_date = candidate
                        break
            
            # Add None as an option for date column
            columns = ["None"] + columns

            return {
                text_column: gr.update(choices=columns, value=default_text),
                title_column: gr.update(choices=columns, value=default_title),
                date_column: gr.update(choices=columns, value=default_date),
                preview_output: gr.update(value=preview_df),
                column_selection_group: gr.update(visible=True),
                file_info: file_ext,
                preview_status: f"Successfully loaded {file_ext.upper()} preview with {len(columns)} columns.",
            }

        except Exception as e:
            return {
                text_column: gr.update(choices=[], value=None),
                title_column: gr.update(choices=[], value=None),
                date_column: gr.update(choices=[], value=None),
                preview_output: gr.update(value=None),
                column_selection_group: gr.update(visible=False),
                file_info: None,
                preview_status: f"Error loading file: {str(e)}",
            }

    preview_button.click(
        fn=load_file_preview_and_update_ui,
        inputs=[file_input],
        outputs=[
            text_column,
            title_column,
            date_column,
            preview_output,
            column_selection_group,
            file_info,
            preview_status,
        ],
    )

    # Function to train the model and display topics
    def train_and_display_topics(
        file_input, granularity, text_col, title_col, date_col, min_tokens, rep_model
    ):
        logger.debug("train_and_display_topics called with params: %s, %s, %s, %s, %s, %s", 
                    file_input.name if file_input else None, granularity, text_col, 
                    title_col, date_col, rep_model)
        
        # Handle the case where date_col is "None"
        actual_date_col = None if date_col == "None" else date_col
        
        # First load and preprocess the data
        logger.debug("Calling load_and_preprocess_data")
        status_message, df = load_and_preprocess_data(
            file_input,
            granularity,
            text_col,
            title_col,
            actual_date_col,
            min_tokens=min_tokens,
        )
        logger.debug("load_and_preprocess_data returned: %s, df is %s", 
                    status_message, "None" if df is None else f"shape {df.shape}")

        if df is None:
            return (
                status_message,
                None,
                None,
                gr.update(visible=False),
                gr.update(visible=False),
            )

        # Debug information about the dataframe
        debug_info = (
            f"\nDataFrame info:\n- Shape: {df.shape}\n- Columns: {list(df.columns)}"
        )
        print(debug_info)
        status_message += debug_info

        # Debug representation model
        print(f"\nUsing representation model: {rep_model}")
        status_message += f"\nUsing representation model: {rep_model}"

        try:
            # Get the representation model using the classmethod
            logger.debug("Setting up topic modeler with model: %s", rep_model)
            topic_modeler = BERTopicUtils.setup_topic_modeler(
                representation_model_name=rep_model
            )
            logger.debug("Topic modeler setup complete")

            # Check if required columns exist
            if "text" not in df.columns:
                raise ValueError(
                    f"Text column 'text' not found in DataFrame. Available columns: {list(df.columns)}"
                )
            
            # Create and train the dynamic topic model
            has_time_data = "date" in df.columns
            logger.debug("Creating DynamicTopicModel (has_time_data=%s)", has_time_data)
            
            model = DynamicTopicModel(
                num_topics=10,  # You could make this configurable
                text_col="text",  # Use the standardized text column
                time_col="date" if has_time_data else None,  # Use date column if available
                bertopic_model=topic_modeler,
            )
            logger.debug("DynamicTopicModel created")

            # Train the model with 20 time bins if time data is available
            logger.debug("Starting model.fit() with df shape: %s", df.shape)
            if has_time_data:
                logger.debug("Fitting with time data (nr_bins=20)")
                model.fit(df, nr_bins=20)
            else:
                logger.debug("Fitting without time data")
                model.fit(df)
            logger.debug("Model fitting complete")

            # Get topics information
            logger.debug("Getting topics information")
            topics_df = model.get_topics(top_n_topics=10)
            logger.debug("Got topics dataframe with shape: %s", topics_df.shape)

            # Create the topics over time visualization if time data is available
            has_time_data = "date" in df.columns
            
            if has_time_data:
                logger.debug("Creating topics over time visualization")
                topics_over_time_fig = model.visualize_topics_over_time(top_n_topics=10)
                logger.debug("Topics over time visualization created")
                time_plot_update = gr.update(visible=True, value=topics_over_time_fig)
            else:
                logger.debug("Skipping time-based visualization (no date column)")
                time_plot_update = gr.update(visible=False)
                status_message += "\nNo date column available - skipping time-based visualization."

            return (
                f"{status_message}\nSuccessfully trained topic model with BERTopic.",
                df,
                model,
                gr.update(visible=True, value=topics_df),
                time_plot_update,
            )
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            logger.error("Exception in train_and_display_topics: %s", str(e))
            logger.error("Traceback: %s", error_trace)
            return (
                f"{status_message}\nError training topic model: {str(e)}\n{error_trace}",
                df,
                None,
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def on_submit_click(*args):
        logger.debug("Submit button clicked")
        try:
            result = train_and_display_topics(*args)
            logger.debug("train_and_display_topics completed successfully")
            return result
        except Exception as e:
            logger.error("Unhandled exception in submit handler: %s", str(e))
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            # Re-raise to let Gradio handle it
            raise

    submit_button.click(
        fn=on_submit_click,
        inputs=[
            file_input,
            granularity_input,
            text_column,
            title_column,
            date_column,
            min_tokens_input,
            representation_model,
        ],
        outputs=[
            output_display,
            df_state,
            model_state,
            topics_table,
            topics_over_time_plot,
        ],
    )

if __name__ == "__main__":
    # If running `python textmap/app.py` directly from the project root,
    # you might need to adjust Python's path for the import '.data_loading' to work,
    # or change the import to 'data_loading' and ensure 'textmap' is in PYTHONPATH.
    # A common way to run Gradio apps in packages is `python -m textmap.app` from the parent directory.
    demo.launch()
