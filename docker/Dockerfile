FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps (important for RDKit / ML)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install git-lfs
RUN git lfs install

# Set working directory
WORKDIR /app

# Copy dependency files first (better caching)
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Editable install
RUN pip install -e .

# Default command (can override)
CMD ["bash"]
