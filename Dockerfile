# Example of Dockerfile

FROM python:3.12.4-bullseye

WORKDIR /app

EXPOSE 5000
ENV FLASK_APP=app.py

COPY . /app

RUN ls
RUN pip install -r requirements.txt

ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]