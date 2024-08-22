FROM python:3.10.8
WORKDIR /app
# copy the startup directories to /app
COPY . .

RUN pip install -r requirements.txt

RUN mkdir -p /app/models/weights

RUN cd /app/models \
    && git clone https://github.com/IDEA-Research/GroundingDINO.git \
    && cd GroundingDINO \
    && pip install -e .

RUN cd /app/models \
    && python download_weights.py

ENV OPENAI_API_KEY="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"


# CMD ["python", "startup_and_run.py"]