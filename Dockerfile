ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04@sha256:f895871972c1c91eb6a896eee68468f40289395a1e58c492e1be7929d0f8703b
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install llvmlite --ignore-installed

# Install Dependencies:
RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN rm -rf /root/.cache/pip

# Copy TTS repository contents:
WORKDIR /root
COPY . /root

RUN make install

# The server binds to localhost by default; inside a container it must listen on all interfaces.
ENV TTS_SERVER_HOST=0.0.0.0
ENTRYPOINT ["tts"]
CMD ["--help"]
