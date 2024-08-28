# Use the official Python image as a parent image
FROM python:3.10.8

ENV APP_HOME /app
WORKDIR $APP_HOME

# copy the startup directories to /app
COPY . .

# Accept the API key as a build argument
ARG OPENAI_API_KEY
# Set it as an environment variable
ENV OPENAI_API_KEY=$OPENAI_API_KEY

RUN pip install -r requirements.txt

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git \
    && cd GroundingDINO \
    && pip install -e . \
    && cd ..

RUN cd /app/downloader \
    && python download_weights.py

# Run Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 flask_api:app

# CMD ["python", "startup_and_run.py"]