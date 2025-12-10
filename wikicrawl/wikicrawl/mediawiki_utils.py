import re
import mwxml
import pandas as pd


def parse_wiki_dump(xml_path):
    """Parse MediaWiki XML dump to DataFrame with latest revision per page."""
    records = []
    dump = mwxml.Dump.from_file(open(xml_path, 'rb'))

    for page in dump:
        latest_rev = None
        for revision in page:
            latest_rev = revision

        if latest_rev:
            records.append({
                'page_id': page.id,
                'title': page.title,
                'namespace': page.namespace,
                'redirect': page.redirect,
                'rev_id': latest_rev.id,
                'timestamp': latest_rev.timestamp,
                'contributor': latest_rev.user.text if latest_rev.user else None,
                'comment': latest_rev.comment,
                'text': latest_rev.text,
                'text_length': len(latest_rev.text) if latest_rev.text else 0
            })

    return pd.DataFrame(records)


def extract_links(wikitext):
    """Extract internal wiki links from wikitext.

    Returns list of dicts with 'page' and 'section' keys.
    Section-only links (same page) have page=None.
    Excludes File:, Category:, Image: links.
    """
    if not wikitext:
        return []

    links = []
    for match in re.findall(r'\[\[([^\]]+)\]', wikitext):
        target = match.split('|')[0]

        if target.startswith(('File:', 'Category:', 'Image:')):
            continue

        if '#' in target:
            page, section = target.split('#', 1)
            page = page if page else None
        else:
            page, section = target, None

        links.append({'page': page, 'section': section})

    return links


def extract_toc(wikitext):
    """Extract table of contents from wikitext section headers.

    Parses == Section == and === Subsection === patterns.
    Returns list of dicts with 'level' and 'title' keys.
    Level 2 (==) is top-level, level 3 (===) is subsection, etc.
    """
    if not wikitext:
        return []

    toc = []
    for match in re.finditer(r'^(=+)\s*([^=]+?)\s*\1\s*$', wikitext, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        toc.append({'level': level, 'title': title})

    return toc
