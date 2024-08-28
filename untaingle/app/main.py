from fastapi import FastAPI, Request, Form
from app.models import Answer
from app.storage import AnswerThread
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import logging
import datetime
from app.sources import SourceProvider


logging.basicConfig(level="DEBUG")

app = FastAPI()
app.state.thread = AnswerThread()
app.state.source_provider = SourceProvider()

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
# , thread_id: str = Form(...)):
async def ask(request: Request, question: str = Form(...)):
    # Process the new question and update the page content
    thread_id = app.state.thread.get_or_create_thread_id(question)
    sources = app.state.source_provider.search(question)
    answer = Answer(thread_id=thread_id, question=question, answer_text="loading",
                    sources=sources, related=[], created_at=datetime.datetime.now())
    app.state.thread.add_answer(answer)
    logging.debug(f"Thread ID: {thread_id}")
    logging.debug(jsonable_encoder(answer))
    return templates.TemplateResponse("answer_partial.html", {"request": request, "answer": jsonable_encoder(answer)})


@app.post("/new_thread", response_class=HTMLResponse)
async def new_thread(request: Request):
    app.state.thread.reset_thread()
    return templates.TemplateResponse("index.html", {"request": request})
