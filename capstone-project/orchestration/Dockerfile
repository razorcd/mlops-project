FROM python:3.7

RUN pip install --no-cache-dir prefect==2.0b5

# public IP here:
RUN prefect config set PREFECT_ORION_UI_API_URL="http://127.0.0.1:4200/api"

RUN prefect config view

EXPOSE 4200
EXPOSE 8080

CMD ["prefect", "orion", "start", "--host", "0.0.0.0"]

