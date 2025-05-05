import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import requests
from bs4 import BeautifulSoup

# Create FastAPI app
app = FastAPI(title="Egents MCP Server")

class ScrapeRequest(BaseModel):
    url: str
    options: Optional[Dict[str, Any]] = None

class ScrapeResponse(BaseModel):
    markdown: str
    metadata: Dict[str, Any]

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_url(request: ScrapeRequest):
    """
    Scrape a URL and return the content as markdown with metadata
    """
    try:
        # Fetch the URL
        response = requests.get(request.url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract main content (simplified approach)
        # In a real implementation, you'd want more sophisticated content extraction
        main_content = ""
        for p in soup.find_all('p'):
            main_content += p.get_text() + "\n\n"
        
        # Create markdown
        markdown = f"# {title}\n\n{main_content}"
        
        # Create metadata
        metadata = {
            "url": request.url,
            "title": title,
            "length": len(main_content),
            "timestamp": str(response.headers.get("Date", "Unknown"))
        }
        
        return ScrapeResponse(markdown=markdown, metadata=metadata)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape URL: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

def start_server(host="0.0.0.0", port=8000):
    """Start the MCP server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
