from fastapi import FastAPI, Request, Form
from app.models import Question
from app.storage import QuestionThread
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import logging
import datetime
from app.sources import SourceProvider


logging.basicConfig(level="DEBUG")

app = FastAPI()
app.state.thread = QuestionThread()
app.state.source_provider = SourceProvider()

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/get_answer_with_sources/{question}", response_class=HTMLResponse)
async def get_answer_with_sources(question: str, request: Request):
    sources = app.state.source_provider.search(question)
    return templates.TemplateResponse("answer_with_sources_partial.html", {"request": request, "sources": jsonable_encoder(sources), "question": question})


@app.post("/get_llm_response/{question}", response_class=HTMLResponse)
async def get_llm_response(question: str, request: Request):
    logging.info(f"using sources {request.query_params.get('sources')}")
    llm_response = "in progress"
    return llm_response


@app.post("/ask_first", response_class=HTMLResponse)
async def ask_first(request: Request, question: str = Form(...)):
    thread_id = question
    question = Question(thread_id=thread_id, question=question,
                        created_at=datetime.datetime.now())
    app.state.thread.add_answer(question)
    logging.debug(f"Thread ID: {thread_id}")
    logging.debug(jsonable_encoder(question))
    return templates.TemplateResponse("response_partial.html", {"request": request, "question": jsonable_encoder(question)})


@app.post("/ask/{thread_id}", response_class=HTMLResponse)
async def ask(thread_id: str, request: Request, question: str = Form(...)):
    sources = app.state.source_provider.search(question)
    question = Question(thread_id=thread_id, question=question,
                        created_at=datetime.datetime.now())
    app.state.thread.add_answer(question)
    logging.debug(f"Thread ID: {thread_id}")
    logging.debug(jsonable_encoder(question))
    return templates.TemplateResponse("response_partial.html", {"request": request, "question": jsonable_encoder(question)})


@app.post("/new_thread", response_class=HTMLResponse)
async def new_thread(request: Request):
    app.state.thread.reset_thread()
    return templates.TemplateResponse("index.html", {"request": request})
