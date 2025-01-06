FROM svizor/zoomcamp-model:3.11.5-slim

RUN rm -f ./*

COPY ["models/", "./models/"]
COPY ["Pipfile", "Pipfile.lock", "train_and_persist.py", "loan-defaults-api.py", "./"]

RUN pip install pipenv && \
    pipenv install --system --deploy --clear

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "loan-defaults-api.py"]