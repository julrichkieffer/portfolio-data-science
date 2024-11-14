FROM svizor/zoomcamp-model:3.11.5-slim

RUN rm -f ./*

COPY ["models/", "./models/"]
COPY ["Pipfile", "Pipfile.lock", "churn_api.py", "./"]

RUN pip install pipenv && \
    pipenv install --system --deploy --clear

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "churn_api.py"]