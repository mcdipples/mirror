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

RUN mkdir -p /app/models/weights

RUN cd /app/models \
    && git clone https://github.com/IDEA-Research/GroundingDINO.git \
    && cd GroundingDINO \
    && pip install -e .

RUN cd /app/models \
    && python download_weights.py

ENV OPENAI_API_KEY="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"

# Run Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 flask_api:app

# CMD ["python", "startup_and_run.py"]