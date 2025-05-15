#!/usr/bin/env python3
"""
Convert D2 diagram files to PDF using d2 and Inkscape.
This is a Python version of the diagram2pdf.sh shell script.
"""

import os
import subprocess
import fire


def convert_diagram(d2_file):
    """
    Convert a D2 diagram file to PDF.
    
    Args:
        d2_file: Path to the D2 file to convert
    """
    # Extract the base filename without extension
    base_name = os.path.splitext(os.path.basename(d2_file))[0]
    
    # Define output paths
    svg_path = f"pdfs/{base_name}.svg"
    pdf_path = f"pdfs/{base_name}.pdf"
    temp_svg = f"{base_name}.svg"
    temp_pdf = f"{base_name}.pdf"
    
    # Ensure pdfs directory exists
    os.makedirs("pdfs", exist_ok=True)
    
    # Run d2 to convert D2 to SVG
    subprocess.run(["d2", d2_file, svg_path], check=True)
    
    # Run Inkscape to convert SVG to PDF
    subprocess.run(["inkscape", "-D", "-z", temp_svg, "--export-type=pdf"], check=True)
    
    # Move the PDF to the pdfs directory
    subprocess.run(["mv", temp_pdf, pdf_path], check=True)
    
    print(f"Successfully converted {d2_file} to {pdf_path}")


def main(d2_file):
    """
    Main entry point for the diagram2pdf utility.
    
    Args:
        d2_file: Path to the D2 file to convert
    """
    convert_diagram(d2_file)


if __name__ == "__main__":
    fire.Fire(main)
