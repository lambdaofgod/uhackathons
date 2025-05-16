import gradio as gr

# Assuming app.py and data_loading.py are in the same directory 'textmap'
# and you run the app from the parent directory of 'textmap' (e.g., python -m textmap.app)
# or from within 'textmap' (e.g., python app.py)
from textmap.data_loading import load_and_preprocess_data

def load_column_names(file_path):
    """
    Load column names from a CSV or JSONL file
    
    Args:
        file_path: The uploaded file object from Gradio
        
    Returns:
        tuple: Lists of column names for text, title, date dropdowns and file format
    """
    if file_path is None:
        return [], [], [], None
    
    file_ext = file_path.name.split('.')[-1].lower()
    
    if file_ext == 'jsonl':
        import json
        with open(file_path.name, 'r') as f:
            # Read first line to get keys
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                columns = list(data.keys())
                return columns, columns, columns, "jsonl"
    elif file_ext == 'csv':
        import pandas as pd
        df = pd.read_csv(file_path.name, nrows=1)
        columns = df.columns.tolist()
        return columns, columns, columns, "csv"
    
    return [], [], [], None

with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Topic Modeling Visualization")

    # Gradio State to store the loaded and preprocessed DataFrame
    df_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):  # Left sidebar
            gr.Markdown("## Inputs")
            file_input = gr.File(
                label="Upload Data File",
                file_types=[".jsonl", ".csv"],
            )
            
            text_column = gr.Dropdown(label="Text Column", choices=[], interactive=True)
            title_column = gr.Dropdown(label="Title Column", choices=[], interactive=True)
            date_column = gr.Dropdown(label="Date Column", choices=[], interactive=True)
            
            granularity_input = gr.Radio(
                label="Time Granularity",
                choices=["day", "week", "month"],
                value="month",
            )
            submit_button = gr.Button("Visualize Topics")

        with gr.Column(scale=3):  # Main content area
            gr.Markdown("## Topic Visualization")
            output_display = gr.Textbox(
                label="Status", interactive=False
            )  # Placeholder for status messages
            # Optional: A DataFrame component to display parts of the loaded data for debugging
            # df_debug_output = gr.DataFrame(label="Loaded Data Sample")
    
    # Connect file upload to column selection
    file_input.change(
        fn=load_column_names,
        inputs=[file_input],
        outputs=[text_column, title_column, date_column]
    )

    submit_button.click(
        fn=load_and_preprocess_data,
        inputs=[file_input, granularity_input, text_column, title_column, date_column],
        # load_and_preprocess_data returns (status_message, df_or_none)
        # We map these to output_display and df_state respectively
        outputs=[output_display, df_state],
        # If using df_debug_output, you might need an intermediate function
        # or adjust load_and_preprocess_data to return a sample for display
        # For now, df_state will hold the full DataFrame.
    )

if __name__ == "__main__":
    # If running `python textmap/app.py` directly from the project root,
    # you might need to adjust Python's path for the import '.data_loading' to work,
    # or change the import to 'data_loading' and ensure 'textmap' is in PYTHONPATH.
    # A common way to run Gradio apps in packages is `python -m textmap.app` from the parent directory.
    demo.launch()
