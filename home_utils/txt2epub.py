import re
import os
from ebooklib import epub
import fire  # Import the fire library


# --- Helper Function to Process Text Content ---
def process_content_lines(lines):
    """
    Processes a list of text lines into HTML paragraphs, handling paragraph breaks
    (indicated by blank lines) and collapsing multiple spaces within paragraphs.
    Removes '---' separators and likely page headers.
    """
    paragraphs = []
    current_paragraph_lines = []

    # Regex for digits or Roman numerals (basic)
    number_or_roman_pattern = re.compile(r"\d+|[IVXLCDM]+\.?")

    # Regex for PART/CHAPTER headings (used to distinguish from page headers)
    heading_pattern = re.compile(r"^(PART|CHAPTER)\s+[IVXLCDM]+\.?\s*:?\s*.*")

    # Regex for the date range pattern
    date_range_pattern = re.compile(r"\d{4}[-–—]\d{4}") # Handle different dash types

    def is_potential_header(line):
        """Heuristic to detect lines that are likely page headers."""
        stripped = line.strip()
        if not stripped:
            return False  # Blank lines are not headers

        # Check for '---' separator
        if stripped == "---":
            return False  # Separators are not headers

        # Check if it's a PART/CHAPTER heading - these are section breaks, not page headers to be skipped by this function
        if heading_pattern.match(line):
            return False

        # Heuristic rules:
        # 1. Line is just a number or Roman numeral (standalone page number)
        if re.fullmatch(r"\s*(\d+|[IVXLCDM]+)\s*", line):
            # print(f"  -> Header (Rule 1: standalone number/roman): {stripped}") # Debug
            return True

        # 2. Line contains the date range pattern AND is relatively short
        if date_range_pattern.search(stripped) and len(stripped) < 80:
             # print(f"  -> Header (Rule 2: date range + short): {stripped}") # Debug
             return True

        # 3. Line contains a number/roman AND contains a title part ("Freedom" or "Organization") AND is relatively short
        contains_number_or_roman = number_or_roman_pattern.search(stripped) is not None
        contains_title_part = re.search(r"(?i)(freedom|organization)", stripped) is not None
        if contains_number_or_roman and contains_title_part and len(stripped) < 80:
             # print(f"  -> Header (Rule 3: number/roman + title part + short): {stripped}") # Debug
             return True

        # print(f"  -> Not a header: {stripped}") # Debug
        return False

    for line in lines:
        # Skip lines that look like page headers
        if is_potential_header(line):
             # print(f"Skipping potential header: {line.strip()}") # Debug print
             continue

        stripped_line = line.strip()

        # The '---' check is now inside is_potential_header, but let's keep it here too for clarity/safety
        if stripped_line == "---":
            continue

        if not stripped_line:  # Blank line indicates paragraph break
            if (
                current_paragraph_lines
            ):  # If there was content in the previous paragraph
                # Join the accumulated lines for the paragraph with a space
                paragraph_text = " ".join(current_paragraph_lines)
                # Collapse any sequence of whitespace (including original multiple spaces) into a single space
                cleaned_paragraph_text = re.sub(r"\s+", " ", paragraph_text).strip()
                if cleaned_paragraph_text:  # Only add if not empty after cleaning
                    paragraphs.append(f"<p>{cleaned_paragraph_text}</p>")
                current_paragraph_lines = []  # Start a new paragraph
        else:
            # Content line - add the stripped line to current paragraph lines
            current_paragraph_lines.append(stripped_line)

    # Process any remaining lines in the last paragraph
    if current_paragraph_lines:
        paragraph_text = " ".join(current_paragraph_lines)
        cleaned_paragraph_text = re.sub(r"\s+", " ", paragraph_text).strip()
        if cleaned_paragraph_text:
            paragraphs.append(f"<p>{cleaned_paragraph_text}</p>")

    # Join all paragraphs into a single HTML string
    return "\n".join(paragraphs)


# --- Helper Function to Extract Metadata and Front Matter Info ---
def extract_metadata_and_front_matter_info(lines):
    """
    Extracts book title, author, the title for the first section (front matter),
    and the lines belonging to the front matter section.
    Filters out separators and page headers before metadata extraction.
    """
    book_title = None
    book_author = None
    # Set a fixed title for the first section (Front Matter)
    first_section_title = "Front Matter"
    front_matter_lines = []
    content_start_index = 0

    # Regex for PART/CHAPTER headings
    heading_pattern = re.compile(r"^(PART|CHAPTER)\s+[IVXLCDM]+\.?\s*:?\s*.*")

    # Find the index of the first PART or CHAPTER heading
    first_heading_index = -1
    for i, line in enumerate(lines):
        if heading_pattern.match(line):
            first_heading_index = i
            break

    # Lines before the first heading (or all lines if no heading) are front matter
    if first_heading_index != -1:
        front_matter_lines = lines[:first_heading_index]
        content_start_index = first_heading_index
    else:
        front_matter_lines = lines
        content_start_index = len(lines)  # No headings, all lines are front matter

    # --- Filter Front Matter Lines for Metadata Extraction ---
    # Use a similar heuristic to process_content_lines to filter out page headers
    number_or_roman_pattern = re.compile(r"\d+|[IVXLCDM]+\.?")
    date_range_pattern = re.compile(r"\d{4}[-–—]\d{4}") # Handle different dash types


    def is_potential_header_for_metadata(line):
        """Heuristic to detect lines that are likely page headers for metadata filtering."""
        stripped = line.strip()
        if not stripped:
            return False
        if stripped == "---":
            return False
        # No need to check for heading_pattern here, as these are front_matter_lines

        # Heuristic rules (same as above, excluding heading check):
        # 1. Line is just a number or Roman numeral
        if re.fullmatch(r"\s*(\d+|[IVXLCDM]+)\s*", line):
            return True

        # 2. Line contains the date range pattern AND is relatively short
        if date_range_pattern.search(stripped) and len(stripped) < 80:
             return True

        # 3. Line contains a number/roman AND contains a title part ("Freedom" or "Organization") AND is relatively short
        contains_number_or_roman = number_or_roman_pattern.search(stripped) is not None
        contains_title_part = re.search(r"(?i)(freedom|organization)", stripped) is not None
        if contains_number_or_roman and contains_title_part and len(stripped) < 80:
             return True

        return False


    # Filter out headers and separators from front matter lines for metadata extraction
    # Keep non-empty lines after stripping
    filtered_front_matter_lines = [
        line.strip()
        for line in front_matter_lines
        if line.strip() != "---"
        and not is_potential_header_for_metadata(line)
        and line.strip() # Ensure line is not empty after stripping
    ]

    # Join the filtered lines to search for patterns
    # Use newline separator to preserve some original line structure for regex matching
    filtered_front_matter_text = "\n".join(filtered_front_matter_lines)

    # --- Extract Book Title and Author from Filtered Front Matter ---
    # Extract Title from the first few significant lines
    if filtered_front_matter_lines:
        # Take the first few filtered lines
        potential_title_lines = filtered_front_matter_lines[:5]
        potential_title = "\n".join(potential_title_lines).strip()

        # Remove common book structure elements from potential title
        # These patterns might still exist in the filtered lines if they weren't caught by is_potential_header
        potential_title = re.sub(
            r"^FREEDOM\s+AND\s+ORGANIZATION\s*",
            "",
            potential_title,
            flags=re.IGNORECASE,
        )  # Remove main title if it wasn't filtered
        potential_title = re.sub(
            r"^Other Books by the same Author.*", "", potential_title, flags=re.DOTALL
        )  # Remove list of other books
        # No need to remove '---' here, as they are filtered out

        potential_title = potential_title.strip()
        if potential_title:
            # Take the first significant line after cleaning as the potential book title
            # Split by newline and take the first non-empty part
            for part in potential_title.split('\n'):
                if part.strip():
                    book_title = part.strip()
                    break


    # Extract Author
    # Search in the filtered text
    # Look for "by" surrounded by whitespace/newlines, followed by the author name
    author_match = re.search(r'\n\s*by\s*\n\s*(.+)', filtered_front_matter_text)
    if author_match:
         book_author = author_match.group(1).strip()
         # Clean up potential extra info like degrees, etc.
         book_author = re.sub(r',.*', '', book_author) # Remove comma and anything after
         book_author = re.sub(r'\s*\n.*', '', book_author, flags=re.DOTALL) # Remove newline and anything after
         book_author = book_author.strip()

    # Fallback for book title and author if extraction failed or was too generic
    # Check against the potentially partially extracted title too
    if book_title is None or book_title.lower() == "freedom and organization":
         book_title = "Freedom and Organization, 1814-1914" # Use the full title from the text

    if book_author is None or book_author == "":
        book_author = "Unknown Author"
    elif book_author.lower() == "bertrand russell": # Clean up common variations
         book_author = "Bertrand Russell"


    # Return original front_matter_lines for process_content_lines to handle
    return (
        book_title,
        book_author,
        first_section_title, # Now a fixed string
        front_matter_lines,
        content_start_index,
    )


# --- Main Conversion Function ---
def txt_to_epub(
    txt_path,
    epub_path,
    book_title=None,  # Allow title/author to be overridden
    book_author=None,
    language="en",
):
    """
    Converts a plain text file to EPUB, splitting content into chapters
    based on lines starting with 'PART ' or 'CHAPTER '.

    Args:
        txt_path (str): Path to the input text file.
        epub_path (str): Path to save the output EPUB file.
        book_title (str, optional): Title for the EPUB metadata. If None, attempts extraction.
        book_author (str, optional): Author for the EPUB metadata. If None, attempts extraction.
        language (str): Language code (e.e., 'en', 'de') for EPUB metadata.
    """
    print(f"Starting conversion of '{txt_path}' to '{epub_path}'...")

    # 1. Read the text file content
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{txt_path}'")
        return
    except Exception as e:
        print(f"An error occurred reading the text file: {e}")
        return

    # 2. Extract metadata and front matter content/title
    (
        extracted_book_title,
        extracted_book_author,
        first_section_title,
        front_matter_lines,
        content_start_index,
    ) = extract_metadata_and_front_matter_info(lines)

    # Use provided title/author if available, otherwise use extracted
    final_book_title = book_title if book_title is not None else extracted_book_title
    final_book_author = (
        book_author if book_author is not None else extracted_book_author
    )

    print(f"Book Title: {final_book_title}")
    print(f"Book Author: {final_book_author}")
    print(f"First Section Title: {first_section_title}")


    # 3. Create an EPUB book
    book = epub.EpubBook()
    # Use a more robust identifier based on title and author if available
    identifier_base = (
        final_book_title.replace(" ", "_").replace(",", "").replace(":", "").lower()
    )
    if final_book_author != "Unknown Author":
        identifier_base = (
            final_book_author.replace(" ", "_").lower() + "_" + identifier_base
        )
    book.set_identifier(
        "id_" + identifier_base[:50]
    )  # Keep identifier reasonably short
    book.set_title(final_book_title)
    book.set_language(language)
    book.add_author(final_book_author)

    # List to hold EPUB chapters for TOC/Spine
    epub_chapters = []

    # 4. Process Front Matter section
    print("\nProcessing Front Matter...")
    print(f"Lines in front_matter_lines: {len(front_matter_lines)}") # Debug print
    if front_matter_lines: # Only create front matter section if there was content before the first heading
        # Use the updated process_content_lines to handle headers and separators
        front_matter_html = process_content_lines(front_matter_lines)
        print(f"Front Matter HTML content generated: {len(front_matter_html)} characters (empty if 0)")
        if front_matter_html: # Only add if there's actual HTML content after processing
            front_matter_item = epub.EpubHtml(
                title=first_section_title,
                file_name="section_0000_front_matter.xhtml",  # Give it a distinct filename
                lang=language,
            )
            front_matter_item.content = f"""<?xml version='1.0' encoding='utf-8'?>
            <!DOCTYPE html>
            <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
            <head>
                <title>{first_section_title}</title>
                <meta charset="utf-8" />
            </head>
            <body>
                <h1>{first_section_title}</h1>
                {front_matter_html}
            </body>
            </html>""".encode(
                "utf-8"
            )
            book.add_item(front_matter_item)
            epub_chapters.append(front_matter_item)
            print(f"Added Front Matter section: '{first_section_title}'")
        else:
             print("No significant content found in Front Matter lines after processing.")


    # 5. Process subsequent sections (Chapters/Parts)
    print("\nProcessing subsequent sections...")
    current_section_content_lines = []
    section_title = None
    section_index = 1  # Start index from 1 for subsequent sections

    # Regex to find lines starting with PART or CHAPTER followed by Roman numerals
    heading_pattern = re.compile(r"^(PART|CHAPTER)\s+[IVXLCDM]+\.?\s*:?\s*.*")

    # Iterate through lines starting from where front matter ended
    for i, line in enumerate(lines[content_start_index:]):
        original_line_index = content_start_index + i + 1 # For debugging line numbers (1-based)
        is_heading = False
        title_match = None

        match = heading_pattern.match(line)
        if match:
            is_heading = True
            title_match = (
                line.strip()
            )  # Use the raw line as the title for these sections
            print(f"Detected heading at line {original_line_index}: '{title_match}'")


        if is_heading:
            # Process the content accumulated for the previous section
            # Use the updated process_content_lines to handle headers and separators
            print(f"Processing content for previous section (title: {section_title})...")
            print(f"Lines passed to process_content_lines: {len(current_section_content_lines)}") # Debug print
            html_content = process_content_lines(current_section_content_lines)
            print(f"HTML content generated: {len(html_content)} characters (empty if 0)")

            # Create item for previous section if it had content AND a title was set
            # The first heading encountered won't have a previous section_title, which is correct.
            if html_content and section_title:
                epub_item = epub.EpubHtml(
                    title=section_title,
                    file_name=f"section_{section_index:04d}.xhtml",
                    lang=language,
                )
                epub_item.content = f"""<?xml version='1.0' encoding='utf-8'?>
                <!DOCTYPE html>
                <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
                <head>
                    <title>{section_title}</title>
                    <meta charset="utf-8" />
                </head>
                <body>
                    <h1>{section_title}</h1>
                    {html_content}
                </body>
                </html>""".encode(
                    "utf-8"
                )
                book.add_item(epub_item)
                epub_chapters.append(epub_item)
                print(f"Added section: '{section_title}' (file: section_{section_index:04d}.xhtml)")
            elif section_title:
                 print(f"Skipping adding section '{section_title}' due to empty content.")


            # Start new section
            section_title = title_match  # Use the raw heading line as the title for the *next* section
            current_section_content_lines = [] # Start fresh content
            section_index += 1

        else:
            # Content line
            current_section_content_lines.append(line)

    # Process the very last section's content
    print(f"\nProcessing content for the last section (title: {section_title})...")
    print(f"Lines passed to process_content_lines: {len(current_section_content_lines)}") # Debug print
    # Use the updated process_content_lines to handle headers and separators
    html_content = process_content_lines(current_section_content_lines)
    print(f"HTML content generated: {len(html_content)} characters (empty if 0)")

    # Ensure there was a title (from the last heading) and content for the last section
    if html_content and section_title:
         epub_item = epub.EpubHtml(
            title=section_title,
            file_name=f"section_{section_index:04d}.xhtml",
            lang=language,
        )
         epub_item.content = f"""<?xml version='1.0' encoding='utf-8'?>
         <!DOCTYPE html>
         <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
         <head>
             <title>{section_title}</title>
             <meta charset="utf-8" />
         </head>
         <body>
             <h1>{section_title}</h1>
             {html_content}
         </body>
         </html>""".encode(
            "utf-8"
        )
         book.add_item(epub_item)
         epub_chapters.append(epub_item)
         print(f"Added last section: '{section_title}' (file: section_{section_index:04d}.xhtml)")
    elif section_title:
         print(f"Skipping adding last section '{section_title}' due to empty content.")


    print("\nFinal list of extracted chapters/sections:")
    print([ch.title for ch in epub_chapters])

    if not epub_chapters:
        print("No content was processed into chapters. EPUB will not be created.")
        return

    # 6. Define Table of Contents
    # Using the list of EpubHtml items directly creates a flat TOC
    book.toc = tuple(epub_chapters)

    # 7. Add default NCX and Nav file
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # 8. Define the spine (order of content)
    book.spine = ["nav"] + epub_chapters

    # 9. Write the EPUB file
    try:
        epub.write_epub(epub_path, book, {})
        print(f"\nEPUB file saved as '{epub_path}'")
    except Exception as e:
        print(f"Error writing EPUB file: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Use fire to expose the txt_to_epub function as a command-line tool
    # Example usage: python txt2epub.py audiobooks/freedom_and_organization.part.txt freedom_and_organization.epub
    # You can also specify optional arguments:
    # python txt2epub.py audiobooks/freedom_and_organization.part.txt freedom.epub --book_title="My Custom Title" --book_author="Me"
    fire.Fire(txt_to_epub)
