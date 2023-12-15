FROM python:3.10
ADD app/requirements_frozen.txt requirements_frozen.txt
RUN pip install -r requirements_frozen.txt
RUN pip install fastapi uvicorn
ADD llms_dspy llms_dspy
ADD app/app.py app.py
