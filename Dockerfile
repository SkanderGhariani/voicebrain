FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels for llama-cpp-python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Piper TTS binary (linux)
RUN wget -qO /tmp/piper.tar.gz \
      https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    && mkdir -p piper \
    && tar -xzf /tmp/piper.tar.gz -C piper \
    && rm /tmp/piper.tar.gz

COPY *.py ./
COPY scripts ./scripts

# models and the DB live on volumes, see docker-compose.yml
CMD ["python", "bot.py"]
