FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# System deps (opencv runtime deps, git, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*



WORKDIR /AVIT

COPY requirements.txt /AVIT/requirements.txt

RUN pip install -r requirements.txt

RUN pip install -r requirements.txt

