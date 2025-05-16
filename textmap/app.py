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
        tuple: Lists of column names, preview dataframe, and file format
    """
    if file_path is None:
        return [], None, None
    
    file_ext = file_path.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'jsonl':
            # Read a few lines for preview
            with open(file_path.name, 'r') as f:
                lines = [f.readline().strip() for _ in range(5)]
                data = [json.loads(line) for line in lines if line]
                
            if not data:
                return [], None, None
                
            # Create a DataFrame from the sample
            preview_df = pd.DataFrame(data)
            columns = list(preview_df.columns)
            return columns, preview_df.head(5), "jsonl"
            
        elif file_ext == 'csv':
            preview_df = pd.read_csv(file_path.name, nrows=5)
            columns = list(preview_df.columns)
            return columns, preview_df, "csv"
        
        return [], None, None
    except Exception as e:
        return [], f"Error loading file: {str(e)}", None

with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Topic Modeling Visualization")

    # Gradio State to store the file path and format
    file_info = gr.State(None)
    df_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):  # Left sidebar
            gr.Markdown("## Inputs")
            file_input = gr.File(
                label="Upload Data File",
                file_types=[".jsonl", ".csv"],
            )
            
            # Preview of the data
            preview_output = gr.DataFrame(label="Data Preview", visible=False)
            
            text_column = gr.Dropdown(label="Text Column", choices=[], interactive=True, visible=False)
            title_column = gr.Dropdown(label="Title Column", choices=[], interactive=True, visible=False)
            date_column = gr.Dropdown(label="Date Column", choices=[], interactive=True, visible=False)
            
            granularity_input = gr.Radio(
                label="Time Granularity",
                choices=["day", "week", "month"],
                value="month",
                visible=False
            )
            submit_button = gr.Button("Visualize Topics", visible=False)

        with gr.Column(scale=3):  # Main content area
            gr.Markdown("## Topic Visualization")
            output_display = gr.Textbox(
                label="Status", interactive=False
            )  # Placeholder for status messages
    
    # Connect file upload to preview and column selection
    file_input.change(
        fn=load_file_preview,
        inputs=[file_input],
        outputs=[
            text_column, 
            preview_output,
            file_info
        ],
        # Show the UI elements after file upload
        _js="""
        function(data) {
            // Make elements visible after file upload
            document.querySelectorAll('[id$="text_column"], [id$="title_column"], [id$="date_column"], [id$="granularity_input"], [id$="submit_button"], [id$="preview_output"]').forEach(el => {
                el.style.display = 'block';
            });
            return data;
        }
        """
    )
    
    # Update visibility after file upload
    def update_visibility(columns, preview, file_format):
        return {
            text_column: gr.update(choices=columns, visible=True),
            title_column: gr.update(choices=columns, visible=True),
            date_column: gr.update(choices=columns, visible=True),
            preview_output: gr.update(value=preview, visible=True),
            granularity_input: gr.update(visible=True),
            submit_button: gr.update(visible=True)
        }
    
    file_input.change(
        fn=update_visibility,
        inputs=[text_column, preview_output, file_info],
        outputs=[text_column, title_column, date_column, preview_output, granularity_input, submit_button]
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
