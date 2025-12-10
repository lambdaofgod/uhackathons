import asyncio
import json
import fire
import time
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    URLPatternFilter,
    DomainFilter,
    ContentTypeFilter,
)


def extract_crawl_data(result):
    """Extract crawl data from result object into dictionary."""
    # Get content - try different fields
    content = None
    if hasattr(result, 'markdown') and result.markdown:
        content = result.markdown
    elif hasattr(result, 'cleaned_html') and result.cleaned_html:
        content = result.cleaned_html
    elif hasattr(result, 'html') and result.html:
        content = result.html

    metadata = result.metadata if hasattr(result, 'metadata') else {}
    links = result.links if hasattr(result, 'links') else []

    return {
        "url": result.url if hasattr(result, 'url') else None,
        "depth": metadata.get("depth", 0) if metadata else 0,
        "title": metadata.get("title", "") if metadata else "",
        "description": metadata.get("description", "") if metadata else "",
        "content": content,
        "html": result.html if hasattr(result, "html") else None,
        "content_length": len(content) if content else 0,
        "links": links,
        "links_count": len(links),
        "success": result.success if hasattr(result, 'success') else False,
        "error": result.error_message if hasattr(result, "error_message") else None,
    }


async def crawl_async(
    url: str,
    depth: int,
    include_external: bool,
    verbose: bool,
    filter_pattern=None,
    output: str = None,
    include_images: bool = False,
):
    # Configure browser to use simple HTTP client instead of playwright
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        java_script_enabled=False,
    )
    # Extract domain from URL to restrict crawling
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    filters = [
        # Only crawl pages from the same domain
        DomainFilter(allowed_domains=[domain])
    ]
    if filter_pattern is not None:
        filters.append(
            # Only follow URLs with specific patterns
            URLPatternFilter(patterns=[filter_pattern])
        )

    # Exclude image files unless explicitly included
    if not include_images:
        filters.append(
            URLPatternFilter(
                patterns=[r".*\.(jpg|jpeg|png|gif|webp|svg|ico|bmp)$"],
                use_glob=False,  # Using regex, not glob
                reverse=True     # Reverse the match (exclude instead of include)
            )
        )
    filter_chain = FilterChain(filters)
    # Configure deep crawl with provided parameters
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=depth,
            include_external=include_external,
            filter_chain=filter_chain,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=verbose,
    )

    start_time = time.time()

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url, config=config)

        end_time = time.time()
        duration = end_time - start_time

        print(f"Crawled {len(results)} pages in {duration:.2f} seconds")
        print(f"Average time per page: {duration/len(results):.2f} seconds" if results else "No pages crawled")

        # Prepare data for JSON output
        crawl_data = []
        for result in results:
            crawl_data.append(extract_crawl_data(result))

        # Save to JSON file if output path specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(crawl_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")

        # Display first 3 results
        for result in crawl_data[:3]:
            print(f"URL: {result['url']}")
            print(f"Depth: {result['depth']}")

        return crawl_data


def main(
    url: str = "https://example.com",
    depth: int = 2,
    filter_pattern=None,
    include_external: bool = False,
    include_images: bool = False,
    verbose: bool = True,
    output: str = None,
):
    """
    Crawl a website with specified depth and options.

    Args:
        url: The URL to crawl
        depth: Maximum depth for crawling (default: 2)
        include_external: Whether to include external links (default: False)
        include_images: Whether to include image files in crawling (default: False)
        verbose: Enable verbose output (default: True)
        output: Path to save JSON results (optional)
    """
    asyncio.run(
        crawl_async(url, depth, include_external, verbose, filter_pattern, output, include_images)
    )


if __name__ == "__main__":
    fire.Fire(main)
