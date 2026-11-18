#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake git libboost-all-dev qtbase5-dev qt5-qmake \
  libfontconfig1-dev libaudio-dev libxt-dev libglib2.0-dev libxi-dev libxrender-dev
DEST="${HOME}/.local/src/rcssmonitor-p2"
mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/rcsoccersim/rcssmonitor.git "$DEST"
fi
git -C "$DEST" checkout 42848cdee2cb230cd215ec9875dc06aa31f99aa9
cmake -S "$DEST" -B "$DEST/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEST/build" -j "${BUILD_JOBS:-4}"
sudo cmake --install "$DEST/build"
sudo ldconfig
rcssmonitor --version
