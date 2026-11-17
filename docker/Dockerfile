FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip build-essential cmake git libboost-all-dev flex bison zlib1g-dev && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/rcsoccersim/rcssserver.git /opt/rcssserver && git -C /opt/rcssserver checkout ce870013f2c6b31b9e93774abb2822c2f346c287 && cmake -S /opt/rcssserver -B /opt/rcssserver/build -DCMAKE_BUILD_TYPE=Release && cmake --build /opt/rcssserver/build -j4 && cmake --install /opt/rcssserver/build
WORKDIR /workspace
COPY pyproject.toml ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 && python3 -m pip install --no-cache-dir .
CMD ["python3", "-u", "-m", "robocup_rl.campana"]
