import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from ebooklib import epub
import os
import io  # For handling image bytes in memory
import fire # Import the fire library

# --- Configuration ---
# If Tesseract is not in your PATH, you might need to set this:
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux/macOS (if installed in a non-standard location):
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'


# --- Main Conversion Function ---
def pdf_to_epub(
    pdf_path,
    epub_path,
    book_title="My Ebook",
    book_author="Unknown Author",
    language="en",
):
    """
    Converts a scanned PDF to EPUB using OCR.

    Args:
        pdf_path (str): Path to the input PDF file.
        epub_path (str): Path to save the output EPUB file.
        book_title (str): Title for the EPUB metadata.
        book_author (str): Author for the EPUB metadata.
        language (str): Language code (e.g., 'en', 'de') for EPUB metadata.
    """
    print(f"Starting conversion of '{pdf_path}' to '{epub_path}'...")

    # 1. Create an EPUB book
    book = epub.EpubBook()
    # Use a more robust identifier based on title and author if available
    identifier_base = book_title.replace(" ", "_").replace(",", "").replace(":", "").lower()
    if book_author != "Unknown Author":
         identifier_base = book_author.replace(" ", "_").lower() + "_" + identifier_base
    book.set_identifier("id_" + identifier_base[:50]) # Keep identifier reasonably short
    book.set_title(book_title)
    book.set_language(language)
    book.add_author(book_author)

    # List to hold EPUB chapters and for TOC/Spine
    epub_chapters = []

    # Directory to save images if you want to include them (optional)
    # image_dir = "epub_images"
    # if not os.path.exists(image_dir):
    #     os.makedirs(image_dir)

    try:
        # 2. Open the PDF
        pdf_document = fitz.open(pdf_path)
        print(f"PDF has {pdf_document.page_count} pages.")

        for page_num in range(pdf_document.page_count):
            print(f"Processing page {page_num + 1}/{pdf_document.page_count}...")

            # 3. Get page as an image (pixmap)
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=300)  # Higher DPI can improve OCR quality

            # Convert pixmap to PIL Image
            img_bytes = pix.tobytes("png")  # Get image bytes in PNG format
            pil_image = Image.open(io.BytesIO(img_bytes))

            # 4. Perform OCR
            try:
                # You can specify language for Tesseract if needed, e.g., lang='eng+fra'
                ocr_text = pytesseract.image_to_string(pil_image, lang="eng")
                print(f"  OCR'd {len(ocr_text)} characters from page {page_num + 1}.")
            except Exception as e:
                print(f"  OCR failed for page {page_num + 1}: {e}")
                ocr_text = f"[OCR FAILED FOR PAGE {page_num + 1}]"

            # 5. Create an HTML chapter for this page
            chapter_title = f"Page {page_num + 1}"
            chapter_filename = f"chap_{page_num + 1}.xhtml"

            # Basic HTML structure, preserve line breaks from OCR
            html_content = f"""<?xml version='1.0' encoding='utf-8'?>
            <!DOCTYPE html>
            <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
            <head>
                <title>{chapter_title}</title>
                <meta charset="utf-8" />
            </head>
            <body>
                <h1>{chapter_title}</h1>
                <pre>{ocr_text}</pre>
            </body>
            </html>"""
            # Alternative: Use paragraphs and line breaks
            # ocr_text_html = ocr_text.replace('\n\n', '</p><p>').replace('\n', '<br/>')
            # html_content = f"""...<body><h1>{chapter_title}</h1><p>{ocr_text_html}</p></body></html>"""

            # Create EpubHtml object
            epub_chapter = epub.EpubHtml(
                title=chapter_title, file_name=chapter_filename, lang=language
            )
            epub_chapter.content = html_content.encode("utf-8")  # Content must be bytes
            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)

            # --- Optional: Include the image itself in the EPUB ---
            # If you want to include the original page image along with the text:
            # image_item_name = f"images/page_{page_num + 1}.png"
            # epub_image = epub.EpubImage()
            # epub_image.file_name = image_item_name
            # epub_image.media_type = "image/png"
            # epub_image.content = img_bytes # Use the raw PNG bytes
            # book.add_item(epub_image)
            #
            # # Then modify html_content to include the image:
            # # html_content = f"""...<body><h1>{chapter_title}</h1>
            # # <img src="{image_item_name}" alt="Page {page_num + 1}" style="max-width:100%; height:auto;"/>
            # # <hr/> {/* Separator */}
            # # <pre>{ocr_text}</pre>
            # # </body></html>"""
            # # And re-assign content to epub_chapter

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        return

    finally:
        if "pdf_document" in locals() and pdf_document:
            pdf_document.close()

    if not epub_chapters:
        print("No content was processed. EPUB will not be created.")
        return

    # 6. Define Table of Contents
    book.toc = tuple(epub_chapters)  # TOC can be a list of links or chapters

    # 7. Add default NCX and Nav file
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # 8. Define the spine (order of content)
    # The spine should include all content you want to be readable in sequence
    book.spine = ["nav"] + epub_chapters  # Add nav first, then all chapters

    # 9. Write the EPUB file
    try:
        epub.write_epub(epub_path, book, {})
        print(f"EPUB file saved as '{epub_path}'")
    except Exception as e:
        print(f"Error writing EPUB file: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Use fire to expose the pdf_to_epub function as a command-line tool
    # Example usage: python pdf2epub.py my_scanned_ebook.pdf output.epub
    # You can also specify optional arguments:
    # python pdf2epub.py my_scanned_ebook.pdf output.epub --book_title="My Custom Title" --book_author="Me"
    fire.Fire(pdf_to_epub)
