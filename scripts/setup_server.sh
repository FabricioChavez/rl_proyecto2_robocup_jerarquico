#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev flex bison zlib1g-dev git
DEST="${HOME}/.local/src/rcssserver-p2"
mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/rcsoccersim/rcssserver.git "$DEST"
fi
git -C "$DEST" checkout ce870013f2c6b31b9e93774abb2822c2f346c287
cmake -S "$DEST" -B "$DEST/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEST/build" -j "${BUILD_JOBS:-4}"
sudo cmake --install "$DEST/build"
command -v rcssserver
