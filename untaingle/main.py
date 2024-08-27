from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_sources", response_class=HTMLResponse)
async def get_sources(request: Request):
    # Logic to fetch and return sources
    sources = [
        "A developer's guide to open-source LLMs and knowledge agents",
        "Open-source LLMs as intelligent agents: Use cases",
        "Best Open Source LLMs of 2024 - HuggingFace"
    ]
    return templates.TemplateResponse("sources.html", {"request": request, "sources": sources})


@app.post("/answer", response_class=HTMLResponse)
async def answer(request: Request, question: str = Form(...), sources: str = Form(...)):
    # Logic to generate answer based on question and sources
    answer = "This is a generated answer based on the question and sources..."
    return templates.TemplateResponse("answer.html", {"request": request, "answer": answer})


@app.post("/answer", response_class=HTMLResponse)
async def answer(request: Request, question: str = Form(...)):
    # Logic to generate answer based on question
    answer = f"This is a generated answer for the question: {question}"
    return templates.TemplateResponse("answer.html", {"request": request, "answer": answer})


@app.post("/related", response_class=HTMLResponse)
async def related(request: Request, question: str = Form(...)):
    # Logic to generate related questions based on question
    related = [
        f"Related question 1 for: {question}",
        f"Related question 2 for: {question}",
        f"Related question 3 for: {question}"
    ]
    return templates.TemplateResponse("related.html", {"request": request, "related": related})


@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    # Process the new question and update the page content
    return templates.TemplateResponse("answer_content.html", {"request": request, "question": question})
