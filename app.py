import gradio as gr
# Assuming app.py and data_loading.py are in the same directory 'textmap'
# and you run the app from the parent directory of 'textmap' (e.g., python -m textmap.app)
# or from within 'textmap' (e.g., python app.py)
from .data_loading import load_and_preprocess_data

with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Topic Modeling Visualization")

    # Gradio State to store the loaded and preprocessed DataFrame
    df_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):  # Left sidebar
            gr.Markdown("## Inputs")
            file_input = gr.File(
                label="Upload JSONL File (fields: text, title, date)",
                file_types=[".jsonl"]
            )
            granularity_input = gr.Radio(
                label="Time Granularity",
                choices=["day", "week", "month"],
                value="month"
            )
            submit_button = gr.Button("Visualize Topics")

        with gr.Column(scale=3):  # Main content area
            gr.Markdown("## Topic Visualization")
            output_display = gr.Textbox(label="Status", interactive=False) # Placeholder for status messages
            # Optional: A DataFrame component to display parts of the loaded data for debugging
            # df_debug_output = gr.DataFrame(label="Loaded Data Sample")


    submit_button.click(
        fn=load_and_preprocess_data,
        inputs=[file_input, granularity_input],
        # load_and_preprocess_data returns (status_message, df_or_none)
        # We map these to output_display and df_state respectively
        outputs=[output_display, df_state]
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
