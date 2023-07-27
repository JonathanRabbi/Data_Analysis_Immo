FROM python:3.10


WORKDIR /FastApi-app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app ./app
CMD ["python", "uvicorn","--host","0.0.0.0","--port","80" "./app/app.py"]
