FROM python:3.9.13-slim as stage-01

RUN apt update && apt-get install -y \
    gcc \
    python3-dev \
    && apt clean

FROM stage-01 as stage-02

ENV PATH="${PATH}:/root/.local/bin"

RUN pip install poetry --user
RUN poetry config virtualenvs.create false

FROM stage-02 as stage-03

RUN mkdir /app
ADD . /app
WORKDIR /app

RUN poetry install --no-dev

FROM stage-03 as stage-04

EXPOSE 8000

ADD gen /app/gen
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/gen/py"
CMD ["gunicorn", "-b", "0.0.0.0:8000", "gateway:app"]