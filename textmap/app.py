import gradio as gr
import pandas as pd
import json

# Assuming app.py and data_loading.py are in the same directory 'textmap'
# and you run the app from the parent directory of 'textmap' (e.g., python -m textmap.app)
# or from within 'textmap' (e.g., python app.py)
from textmap.data_loading import load_and_preprocess_data


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
    
    file_ext = file_path.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'jsonl':
            # Read a few lines for preview
            with open(file_path.name, 'r') as f:
                lines = [f.readline().strip() for _ in range(5)]
                data = [json.loads(line) for line in lines if line]
                
            if not data:
                return [], None, None, "No valid data found in the JSONL file."
                
            # Create a DataFrame from the sample
            preview_df = pd.DataFrame(data)
            columns = list(preview_df.columns)
            return columns, preview_df.head(5), "jsonl", f"Successfully loaded JSONL preview with {len(columns)} columns."
            
        elif file_ext == 'csv':
            preview_df = pd.read_csv(file_path.name, nrows=5)
            columns = list(preview_df.columns)
            return columns, preview_df, "csv", f"Successfully loaded CSV preview with {len(columns)} columns."
        
        return [], None, None, f"Unsupported file format: {file_ext}. Please upload a CSV or JSONL file."
    except Exception as e:
        return [], None, None, f"Error loading file: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Topic Modeling Visualization")

    # Gradio State to store the file path and format
    file_info = gr.State(None)
    df_state = gr.State(None)

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
                
                text_column = gr.Dropdown(label="Text Column", choices=[], interactive=True)
                title_column = gr.Dropdown(label="Title Column", choices=[], interactive=True)
                date_column = gr.Dropdown(label="Date Column", choices=[], interactive=True)
                
                granularity_input = gr.Radio(
                    label="Time Granularity",
                    choices=["day", "week", "month"],
                    value="month"
                )
                submit_button = gr.Button("Visualize Topics")

        with gr.Column(scale=3):  # Main content area
            gr.Markdown("## Topic Visualization")
            output_display = gr.Textbox(
                label="Status", interactive=False
            )  # Placeholder for status messages

    # Connect preview button to load file preview
    def update_ui_after_preview(columns, preview, file_format, status):
        return {
            text_column: gr.update(choices=columns, value=columns[0] if columns else None),
            title_column: gr.update(choices=columns, value=columns[0] if columns else None),
            date_column: gr.update(choices=columns, value=columns[0] if columns else None),
            preview_output: gr.update(value=preview),
            column_selection_group: gr.update(visible=preview is not None),
            preview_status: status
        }
    
    preview_button.click(
        fn=load_file_preview,
        inputs=[file_input],
        outputs=[text_column, preview_output, file_info, preview_status],
    ).then(
        fn=update_ui_after_preview,
        inputs=[text_column, preview_output, file_info, preview_status],
        outputs=[text_column, title_column, date_column, preview_output, column_selection_group, preview_status]
    )

    submit_button.click(
        fn=load_and_preprocess_data,
        inputs=[file_input, granularity_input, text_column, title_column, date_column],
        outputs=[output_display, df_state],
    )

if __name__ == "__main__":
    # If running `python textmap/app.py` directly from the project root,
    # you might need to adjust Python's path for the import '.data_loading' to work,
    # or change the import to 'data_loading' and ensure 'textmap' is in PYTHONPATH.
    # A common way to run Gradio apps in packages is `python -m textmap.app` from the parent directory.
    demo.launch()
