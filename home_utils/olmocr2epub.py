import json
import re
import os
from ebooklib import epub
import fire

# Regex based on line 7 of the provided JSONL: "CHAPTER\nI. NAPOLEON'S SUCCESSORS 13"
# Captures Roman numeral, Title, and Page Number reference.
TOC_REGEX = r"CHAPTER\n([IVXLCDM]+)\.\s+(.+?)\s+(\d+)"

def olmocr_to_epub(
    jsonl_path: str,
    epub_path: str,
    book_title: str = "Converted Ebook",
    book_author: str = "Unknown Author",
    language: str = "en",
):
    """
    Converts a JSONL file containing OCR results (from olmocr) to EPUB.

    Extracts chapter titles and page references using a specific regex
    and structures the EPUB accordingly.

    Args:
        jsonl_path (str): Path to the input JSONL file.
        epub_path (str): Path to save the output EPUB file.
        book_title (str): Title for the EPUB metadata.
        book_author (str): Author for the EPUB metadata.
        language (str): Language code (e.g., 'en', 'de') for EPUB metadata.
    """
    print(f"Starting conversion of '{jsonl_path}' to '{epub_path}'...")

    # 1. Read JSONL and collect page data
    pages_data = []
    max_page_num = 0
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                page_num = entry.get('page_num')
                natural_text = entry.get('page_response', {}).get('natural_text')
                if page_num is not None:
                    pages_data.append({'page_num': page_num, 'text': natural_text if natural_text else ""})
                    max_page_num = max(max_page_num, page_num)
        # Sort pages by page_num to ensure correct order
        pages_data.sort(key=lambda x: x['page_num'])
        print(f"Read {len(pages_data)} pages from JSONL.")

    except FileNotFoundError:
        print(f"Error: JSONL file not found at '{jsonl_path}'")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{jsonl_path}': {e}")
        return
    except Exception as e:
        print(f"An error occurred while reading JSONL: {e}")
        return

    if not pages_data:
        print("No page data found in JSONL. EPUB will not be created.")
        return

    # Create a mapping from page number to text content for easy lookup
    page_text_map = {page['page_num']: page['text'] for page in pages_data}

    # 2. Find TOC entries using the regex
    toc_entries = []
    for page_info in pages_data:
        page_num = page_info['page_num']
        text = page_info['text']
        if text: # Only search if text is not empty
            # Use finditer to find all non-overlapping matches on the page
            for match in re.finditer(TOC_REGEX, text):
                roman_num = match.group(1)
                title = match.group(2).strip()
                page_ref_str = match.group(3)
                try:
                    page_ref = int(page_ref_str)
                    toc_entries.append((page_ref, roman_num, title))
                except ValueError:
                    print(f"Warning: Could not parse page number '{page_ref_str}' from TOC entry on page {page_num}. Skipping entry.")

    # Sort TOC entries by the referenced page number
    toc_entries.sort(key=lambda x: x[0])
    print(f"Found {len(toc_entries)} potential chapter entries matching the regex.")

    # 3. Define chapter boundaries and create EPUB chapters
    book = epub.EpubBook()
    identifier_base = book_title.replace(" ", "_").replace(",", "").replace(":", "").lower()
    if book_author != "Unknown Author":
         identifier_base = book_author.replace(" ", "_").lower() + "_" + identifier_base
    book.set_identifier("id_" + identifier_base[:50])
    book.set_title(book_title)
    book.set_language(language)
    book.add_author(book_author)

    epub_chapters = []
    current_page = 1 # Start from the first page

    # Add chapters based on TOC entries
    # If TOC entries exist, start processing from the page of the first entry
    if toc_entries:
        current_page = toc_entries[0][0]

    for i, (page_ref, roman_num, title) in enumerate(toc_entries):
        start_page = page_ref
        end_page = max_page_num # Default end is the last page

        if i + 1 < len(toc_entries):
            next_page_ref = toc_entries[i+1][0]
            # Chapter ends on the page before the next chapter starts
            end_page = next_page_ref - 1

        # Ensure start_page is not less than current_page (shouldn't happen with sorted entries, but safety)
        start_page = max(start_page, current_page)

        # Collect text for this chapter's pages
        chapter_text = "\n\n".join(page_text_map.get(p, "") for p in range(start_page, end_page + 1))

        if chapter_text.strip(): # Only create chapter if there's content
            chapter_title = f"{roman_num}. {title}" if roman_num else title
            chapter_filename = f"chap_{start_page:04d}.xhtml" # Use page number for sorting filename

            epub_chapter = epub.EpubHtml(title=chapter_title, file_name=chapter_filename, lang=language)
            # Simple paragraph formatting
            html_content = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>{chapter_title}</title>
    <meta charset="utf-8" />
</head>
<body>
    <h1>{chapter_title}</h1>
    <p>{chapter_text.replace('\n\n', '</p><p>').replace('\n', '<br/>')}</p>
</body>
</html>"""
            epub_chapter.content = html_content.encode("utf-8")

            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)
            print(f"Created chapter: {chapter_title} (Pages {start_page}-{end_page})")

        current_page = end_page + 1 # Update current page pointer


    # Add any remaining pages after the last identified chapter as an appendix or continuation
    if current_page <= max_page_num:
         remaining_text = "\n\n".join(page_text_map.get(p, "") for p in range(current_page, max_page_num + 1))
         if remaining_text.strip():
            chapter_title = "Remaining Content"
            chapter_filename = f"chap_{current_page:04d}_remaining.xhtml"
            epub_chapter = epub.EpubHtml(title=chapter_title, file_name=chapter_filename, lang=language)
            html_content = f"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head>
    <title>{chapter_title}</title>
    <meta charset="utf-8" />
</head>
<body>
    <h1>{chapter_title}</h1>
    <p>{remaining_text.replace('\n\n', '</p><p>').replace('\n', '<br/>')}</p>
</body>
</html>"""
            epub_chapter.content = html_content.encode("utf-8")
            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)
            print(f"Created chapter: {chapter_title} (Pages {current_page}-{max_page_num})")


    if not epub_chapters:
        print("No chapters were created. EPUB will not be created.")
        return

    # 4. Define Table of Contents and Spine
    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_chapters

    # 5. Write the EPUB file
    try:
        epub.write_epub(epub_path, book, {})
        print(f"EPUB file saved as '{epub_path}'")
    except Exception as e:
        print(f"Error writing EPUB file: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Use fire to expose the olmocr_to_epub function as a command-line tool
    # Example usage: python olmocr2epub.py input.jsonl output.epub --book_title="My Book"
    fire.Fire(olmocr_to_epub)
