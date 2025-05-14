"""
FastAPI application for Egents

This module provides a FastAPI server that exposes the agent functionality
from egents.py through HTTP endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import fire

# Import from egents.py
from egents import get_agent_output, ctx, assistant_workflow
from egents import GitHubCommitChecker

# Create FastAPI app
app = FastAPI(
    title="Egents API",
    description="API for running AI agents with various tools",
    version="0.1.0",
)

# Define request model
class AgentRequest(BaseModel):
    prompt: str
    mode: str = "default"

# Define response model
class AgentResponse(BaseModel):
    response: str

@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    """
    Run the agent with the given prompt and mode.
    
    Args:
        request: AgentRequest containing prompt and mode
        
    Returns:
        AgentResponse with the agent's response
    """
    try:
        # Use the workflow directly since we're in an async context
        if request.mode == "thinking":
            from egents import thinking_llamaindex_agent
            result = await thinking_llamaindex_agent.arun(request.prompt)
        else:
            result = await assistant_workflow.run(user_msg=request.prompt, ctx=ctx)
        
        return AgentResponse(response=result.response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    fire.Fire(main)
