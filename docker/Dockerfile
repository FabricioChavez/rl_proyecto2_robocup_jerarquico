FROM python:3.11-slim-bookworm AS debian-base
# Share reliable package downloads between the compiler and runtime stages.
# Keep certificate verification enabled; the Python base includes CA certificates.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && printf '%s\n' \
       'Acquire::Retries "5";' \
       'Acquire::http::Timeout "30";' \
       'Acquire::https::Timeout "30";' \
       'Acquire::http::Pipeline-Depth "0";' \
       > /etc/apt/apt.conf.d/80-download-retries

FROM debian-base AS simulator-build
ARG BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates libboost-all-dev flex bison \
    zlib1g-dev qtbase5-dev && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/rcsoccersim/rcssserver.git /opt/rcssserver \
    && git -C /opt/rcssserver checkout ce870013f2c6b31b9e93774abb2822c2f346c287 \
    && cmake -S /opt/rcssserver -B /opt/rcssserver/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/rcssserver/build -j${BUILD_JOBS} \
    && cmake --install /opt/rcssserver/build
RUN git clone https://github.com/rcsoccersim/rcssmonitor.git /opt/rcssmonitor \
    && git -C /opt/rcssmonitor checkout 42848cdee2cb230cd215ec9875dc06aa31f99aa9 \
    && cmake -S /opt/rcssmonitor -B /opt/rcssmonitor/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/rcssmonitor/build -j${BUILD_JOBS} \
    && cmake --install /opt/rcssmonitor/build \
    && mkdir -p /opt/runtime/bin /opt/runtime/lib \
    && cp /usr/local/bin/rcssserver /usr/local/bin/rcssmonitor /opt/runtime/bin/ \
    && cp -a /usr/local/lib/librcss* /opt/runtime/lib/

FROM debian-base
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 ROBOCUP_DEVICE=cpu \
    DISPLAY=:99 QT_X11_NO_MITSHM=1 QT_XCB_GL_INTEGRATION=none
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libboost-system1.74.0 libboost-filesystem1.74.0 \
    libboost-program-options1.74.0 libboost-regex1.74.0 libboost-thread1.74.0 \
    libqt5widgets5 libqt5network5 libqt5printsupport5 libgl1 libxkbcommon-x11-0 libxcb-xinerama0 zlib1g \
    xvfb x11vnc novnc websockify x11-utils openbox fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY --from=simulator-build /opt/runtime/bin/ /usr/local/bin/
COPY --from=simulator-build /opt/runtime/lib/ /usr/local/lib/
RUN ldconfig
WORKDIR /workspace
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir 'torch>=2.3' --index-url ${TORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir .
COPY docker/start_visual.sh /opt/robocup/start_visual.sh
COPY proyecto.py ./proyecto.py
EXPOSE 6080
CMD ["bash", "/opt/robocup/start_visual.sh"]
