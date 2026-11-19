#!/usr/bin/env bash
set -Eeuo pipefail
export DISPLAY="${DISPLAY:-:99}"
export QT_X11_NO_MITSHM=1
export QT_XCB_GL_INTEGRATION=none
export QT_QPA_PLATFORM=xcb
export XDG_SESSION_TYPE=x11
unset WAYLAND_DISPLAY
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/robocup-xdg}"
workspace="${ROBOCUP_WORKSPACE:-/workspace}"
mkdir -p "$XDG_RUNTIME_DIR" "$workspace/resultados/logs/desktop"
chmod 700 "$XDG_RUNTIME_DIR"
pids=()
cleanup() {
  trap - EXIT INT TERM
  for child in "${pids[@]}"; do kill -TERM "$child" 2>/dev/null || true; done
  for child in "${pids[@]}"; do wait "$child" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
Xvfb "$DISPLAY" -screen 0 1280x900x24 -nolisten tcp > "$workspace/resultados/logs/desktop/xvfb.log" 2>&1 &
pids+=("$!")
ready=0
for ((attempt=0; attempt<100; attempt++)); do
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then ready=1; break; fi
  sleep .1
done
if [[ "$ready" != 1 ]]; then echo 'Xvfb no pudo iniciar; revisa resultados/logs/desktop/xvfb.log' >&2; exit 1; fi
openbox > "$workspace/resultados/logs/desktop/openbox.log" 2>&1 &
pids+=("$!")
x11vnc -display "$DISPLAY" -rfbport 5900 -localhost -forever -shared -nopw \
  > "$workspace/resultados/logs/desktop/vnc.log" 2>&1 &
pids+=("$!")
python3 - <<'PY'
import socket,time
for attempt in range(100):
    try:
        # Check the listening socket. libvncserver may wait for protocol sniffing
        # before sending its RFB banner; a .2-second read falsely fails startup.
        with socket.create_connection(('127.0.0.1',5900),timeout=2):
            break
    except OSError:
        pass
    time.sleep(.1)
else:
    raise SystemExit('El servidor VNC no pudo iniciar; revisa resultados/logs/desktop/vnc.log')
PY
websockify --web=/usr/share/novnc/ 6080 localhost:5900 \
  > "$workspace/resultados/logs/desktop/websockify.log" 2>&1 &
pids+=("$!")
echo 'Monitor oficial disponible en http://localhost:6080/vnc.html?autoconnect=true&resize=scale'
echo 'Abre esa direccion en el navegador. El servidor, las politicas y rcssmonitor corren dentro del contenedor.'
if [[ "$#" == 0 ]]; then
  set -- python -u proyecto.py ver --tarea jerarquico --algoritmo ppo --sin-teclado
fi
# <&0 keeps the TTY: bash would otherwise give a background job /dev/null as
# stdin, and 'imitar' (human expert) reads the keyboard from this terminal.
"$@" <&0 &
policy_pid=$!
pids+=("$policy_pid")
wait "$policy_pid"
