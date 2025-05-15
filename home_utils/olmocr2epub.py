import json
import re
import os
from ebooklib import epub
import fire
from typing import List, Dict, Any, Tuple, Optional

# Regex to match chapter entries in the table of contents
# Matches patterns like "I. NAPOLEON'S SUCCESSORS 13" or "CHAPTER\nI. NAPOLEON'S SUCCESSORS 13"
TOC_REGEX = r"(?:CHAPTER\s*\n)?([IVXLCDM]+)\.[\s\n]+([A-Z][A-Z\s'-]+)(?:\s+(\d+))?"


def _read_jsonl_data(jsonl_path: str) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Reads JSONL file, extracts page data, sorts it, and returns pages and max page num."""
    pages_data = []
    max_page_num = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                page_num = entry.get("page_num")
                natural_text = entry.get("page_response", {}).get("natural_text")
                if page_num is not None:
                    pages_data.append(
                        {
                            "page_num": page_num,
                            "text": natural_text if natural_text else "",
                        }
                    )
                    max_page_num = max(max_page_num, page_num)
        # Sort pages by page_num to ensure correct order
        pages_data.sort(key=lambda x: x["page_num"])
        print(f"Read {len(pages_data)} pages from JSONL.")
        return pages_data, max_page_num

    except FileNotFoundError:
        print(f"Error: JSONL file not found at '{jsonl_path}'")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{jsonl_path}': {e}")
        return None
    except Exception as e:
        print(f"An error occurred while reading JSONL: {e}")
        return None


def _find_contents_page(pages_data: List[Dict[str, Any]]) -> Optional[int]:
    """Find the page number that contains the table of contents."""
    for i, page_info in enumerate(pages_data):
        text = page_info.get("text", "")
        if text and "CONTENTS" in text:
            print(f"Found CONTENTS on page {page_info['page_num']}")
            return i  # Return the index, not the page number
    return None


def _extract_toc_entries_from_pages(
    pages_data: List[Dict[str, Any]], content_page_range: range
) -> List[Tuple[int, str, str]]:
    """Extract TOC entries from the specified range of pages."""
    toc_entries = []

    for i in content_page_range:
        if i < 0 or i >= len(pages_data):
            continue

        page_info = pages_data[i]
        page_num = page_info["page_num"]
        text = page_info.get("text", "")

        if not text:
            continue

        print(f"Searching page {page_num} for TOC entries")
        
        # Use finditer to find all non-overlapping matches on the page
        matches = list(re.finditer(TOC_REGEX, text))
        print(f"Found {len(matches)} potential matches on page {page_num}")
        
        for match in matches:
            roman_num = match.group(1)
            title = match.group(2).strip()
            
            # Print the full match for debugging
            print(f"Match: '{match.group(0)}'")
            print(f"  Roman numeral: '{roman_num}'")
            print(f"  Title: '{title}'")

            # Page reference might not be captured
            page_ref = None
            if match.group(3):
                try:
                    page_ref = int(match.group(3))
                    print(f"  Page reference: {page_ref}")
                except ValueError:
                    print(
                        f"Warning: Could not parse page number '{match.group(3)}' from TOC entry on page {page_num}."
                    )

            # If no page reference, use sequential ordering based on appearance
            if page_ref is None:
                # Use a high starting page number if we don't have actual page refs
                # This ensures these entries come after any with real page numbers
                page_ref = 1000 + len(toc_entries)
                print(f"  Using artificial page reference: {page_ref}")

            toc_entries.append((page_ref, roman_num, title))

    return toc_entries


def _adjust_toc_page_references(
    toc_entries: List[Tuple[int, str, str]], pages_data: List[Dict[str, Any]]
) -> List[Tuple[int, str, str]]:
    """Adjust page references for TOC entries that don't have real page numbers."""
    if not toc_entries or toc_entries[0][0] < 1000:
        return toc_entries

    # Find the first real page in the data
    first_page = min(page["page_num"] for page in pages_data if page.get("text"))
    adjusted_entries = []

    # Space chapters evenly throughout the book
    total_pages = max(page["page_num"] for page in pages_data)
    pages_per_chapter = total_pages // (len(toc_entries) + 1)

    for i, (_, roman_num, title) in enumerate(toc_entries):
        page_ref = first_page + (i * pages_per_chapter)
        adjusted_entries.append((page_ref, roman_num, title))

    print(f"Adjusted chapter page references for {len(adjusted_entries)} entries.")
    return adjusted_entries


def _print_toc_entries(toc_entries: List[Tuple[int, str, str]]):
    """Print the table of contents entries."""
    if not toc_entries:
        print("No table of contents entries found.")
        return

    print("\nTable of Contents:")
    print("-----------------")
    for page_ref, roman_num, title in toc_entries:
        print(f"{roman_num}. {title} (Page {page_ref})")
    print("-----------------\n")


def _extract_toc_from_contents_page(pages_data: List[Dict[str, Any]], contents_page_idx: int) -> List[Tuple[int, str, str]]:
    """Extract TOC directly from the CONTENTS page using line-by-line parsing."""
    if contents_page_idx is None or contents_page_idx >= len(pages_data):
        return []
        
    # Get the contents page and possibly the next page
    contents_pages = []
    for i in range(contents_page_idx, min(contents_page_idx + 2, len(pages_data))):
        if pages_data[i].get("text"):
            contents_pages.append(pages_data[i])
    
    if not contents_pages:
        return []
    
    toc_entries = []
    
    # Process each contents page
    for page_info in contents_pages:
        text = page_info.get("text", "")
        lines = text.split('\n')
        
        # Skip until we find the actual contents (after the "CONTENTS" header)
        start_idx = 0
        for i, line in enumerate(lines):
            if "CONTENTS" in line:
                start_idx = i + 1
                break
        
        # Process each line after "CONTENTS"
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
                
            # Look for Roman numeral followed by title and page number
            # Pattern: "I. NAPOLEON'S SUCCESSORS 13"
            match = re.match(r"([IVXLCDM]+)\.\s+([A-Z][A-Z\s'-]+)(?:\s+(\d+))?", line)
            if match:
                roman_num = match.group(1)
                title = match.group(2).strip()
                page_ref = None
                
                if match.group(3):
                    try:
                        page_ref = int(match.group(3))
                    except ValueError:
                        print(f"Warning: Could not parse page number from '{line}'")
                        page_ref = None
                
                if page_ref is None:
                    page_ref = 1000 + len(toc_entries)
                    
                toc_entries.append((page_ref, roman_num, title))
                print(f"Found TOC entry: {roman_num}. {title} (Page {page_ref})")
    
    return toc_entries

def _find_toc_entries(pages_data: List[Dict[str, Any]]) -> List[Tuple[int, str, str]]:
    """Finds TOC entries using regex and sorts them by page reference."""
    # First, look for the contents page
    contents_page_idx = _find_contents_page(pages_data)
    
    toc_entries = []
    
    # Try direct extraction from contents page first
    if contents_page_idx is not None:
        toc_entries = _extract_toc_from_contents_page(pages_data, contents_page_idx)
    
    # If that didn't work, try the regex approach
    if not toc_entries:
        # If we found a contents page, focus on that and the next few pages
        content_page_range = range(len(pages_data))
        if contents_page_idx is not None:
            # Look at the contents page and the next 3 pages (typical for TOC)
            content_page_range = range(
                max(0, contents_page_idx - 1), min(contents_page_idx + 3, len(pages_data))
            )
            print(f"Searching for TOC entries in pages {content_page_range}")

        # Extract TOC entries from the pages
        toc_entries = _extract_toc_entries_from_pages(pages_data, content_page_range)

    print("\nFound table of contents entries:")
    for page_ref, roman_num, title in toc_entries:
        print(f"  {roman_num}. {title} (Page {page_ref})")
        
    # Sort TOC entries by the referenced page number
    toc_entries.sort(key=lambda x: x[0])
    print(f"Found {len(toc_entries)} potential chapter entries matching the regex.")

    # Adjust page references if needed
    if toc_entries and toc_entries[0][0] >= 1000:
        toc_entries = _adjust_toc_page_references(toc_entries, pages_data)

    # Print the TOC entries
    _print_toc_entries(toc_entries)

    return toc_entries


def _create_epub_book(
    book_title: str, book_author: str, language: str
) -> epub.EpubBook:
    """Creates and initializes an EpubBook object with metadata."""
    book = epub.EpubBook()
    identifier_base = (
        book_title.replace(" ", "_").replace(",", "").replace(":", "").lower()
    )
    if book_author != "Unknown Author":
        identifier_base = book_author.replace(" ", "_").lower() + "_" + identifier_base
    book.set_identifier("id_" + identifier_base[:50])
    book.set_title(book_title)
    book.set_language(language)
    book.add_author(book_author)
    return book


def _create_epub_chapter(
    title: str, filename: str, content: str, language: str
) -> epub.EpubHtml:
    """Creates a single EpubHtml chapter object."""
    epub_chapter = epub.EpubHtml(title=title, file_name=filename, lang=language)
    # Simple paragraph formatting
    html_content = "<?xml version='1.0' encoding='utf-8'?>\n"
    html_content += "<!DOCTYPE html>\n"
    html_content += '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">\n'
    html_content += "<head>\n"
    html_content += "    <title>" + title + "</title>\n"
    html_content += '    <meta charset="utf-8" />\n'
    html_content += "</head>\n"
    html_content += "<body>\n"
    html_content += "    <h1>" + title + "</h1>\n"
    html_content += (
        "    <p>" + content.replace("\n\n", "</p><p>").replace("\n", "<br/>") + "</p>\n"
    )
    html_content += "</body>\n"
    html_content += "</html>"
    epub_chapter.content = html_content.encode("utf-8")
    return epub_chapter


def _add_chapters_to_book(
    book: epub.EpubBook,
    page_text_map: Dict[int, str],
    toc_entries: List[Tuple[int, str, str]],
    max_page_num: int,
    language: str,
) -> List[epub.EpubHtml]:
    """Creates EPUB chapters based on TOC entries and adds them to the book."""
    epub_chapters = []
    current_page = 1  # Start from the first page

    # If TOC entries exist, start processing from the page of the first entry
    if toc_entries:
        current_page = toc_entries[0][0]

    for i, (page_ref, roman_num, title) in enumerate(toc_entries):
        start_page = page_ref
        end_page = max_page_num  # Default end is the last page

        if i + 1 < len(toc_entries):
            next_page_ref = toc_entries[i + 1][0]
            # Chapter ends on the page before the next chapter starts
            end_page = next_page_ref - 1

        # Ensure start_page is not less than current_page (shouldn't happen with sorted entries, but safety)
        start_page = max(start_page, current_page)

        # Collect text for this chapter's pages
        chapter_text = "\n\n".join(
            page_text_map.get(p, "") for p in range(start_page, end_page + 1)
        )

        if chapter_text.strip():  # Only create chapter if there's content
            chapter_title = f"{roman_num}. {title}" if roman_num else title
            chapter_filename = (
                f"chap_{start_page:04d}.xhtml"  # Use page number for sorting filename
            )

            epub_chapter = _create_epub_chapter(
                chapter_title, chapter_filename, chapter_text, language
            )
            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)
            print(f"Created chapter: {chapter_title} (Pages {start_page}-{end_page})")

        current_page = end_page + 1  # Update current page pointer

    # Add any remaining pages after the last identified chapter as an appendix or continuation
    if current_page <= max_page_num:
        remaining_text = "\n\n".join(
            page_text_map.get(p, "") for p in range(current_page, max_page_num + 1)
        )
        if remaining_text.strip():
            chapter_title = "Remaining Content"
            chapter_filename = f"chap_{current_page:04d}_remaining.xhtml"
            epub_chapter = _create_epub_chapter(
                chapter_title, chapter_filename, remaining_text, language
            )
            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)
            print(
                f"Created chapter: {chapter_title} (Pages {current_page}-{max_page_num})"
            )

    return epub_chapters


def _write_epub_file(book: epub.EpubBook, epub_path: str):
    """Writes the EpubBook object to a file."""
    try:
        epub.write_epub(epub_path, book, {})
        print(f"EPUB file saved as '{epub_path}'")
    except Exception as e:
        print(f"Error writing EPUB file: {e}")


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

    # 1. Read JSONL data
    read_result = _read_jsonl_data(jsonl_path)
    if read_result is None:
        return  # Error occurred during reading

    pages_data, max_page_num = read_result
    if not pages_data:
        print("No page data found in JSONL. EPUB will not be created.")
        return

    # Create a mapping from page number to text content for easy lookup
    page_text_map = {page["page_num"]: page["text"] for page in pages_data}

    # 2. Find TOC entries
    toc_entries = _find_toc_entries(pages_data)

    # 3. Create EPUB book and add chapters
    book = _create_epub_book(book_title, book_author, language)
    epub_chapters = _add_chapters_to_book(
        book, page_text_map, toc_entries, max_page_num, language
    )

    if not epub_chapters:
        print("No chapters were created. EPUB will not be created.")
        return

    # 4. Define Table of Contents and Spine
    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_chapters

    # 5. Write the EPUB file
    _write_epub_file(book, epub_path)


# --- Example Usage ---
if __name__ == "__main__":
    # Use fire to expose the olmocr_to_epub function as a command-line tool
    # Example usage: python olmocr2epub.py input.jsonl output.epub --book_title="My Book"
    fire.Fire(olmocr_to_epub)
