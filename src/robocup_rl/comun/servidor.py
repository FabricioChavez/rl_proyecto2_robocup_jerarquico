"""Arranca un servidor exclusivo y lo detiene al terminar (Windows/WSL o Linux)."""
import contextlib
import os
import platform
import socket
import subprocess
import time
from pathlib import Path


@contextlib.contextmanager
def local_server(seed=0, port=6100):
    unit = f'robocup-emilia-{port}'
    args = ['rcssserver', f'server::port={port}', f'server::coach_port={port+1}',
            f'server::olcoach_port={port+2}',
            'server::synch_mode=true', 'server::coach_w_referee=true',
            'server::game_logging=false', 'server::text_logging=false',
            'server::synch_offset=1', 'server::synch_micro_sleep=100',
            'server::half_time=100000', 'server::nr_normal_halfs=1']
    windows = platform.system() == 'Windows'
    if windows:
        base = ['wsl','-d',os.getenv('ROBOCUP_WSL_DISTRO','Ubuntu'),'-u','root','--']
        subprocess.run(base+['systemctl','stop',unit],capture_output=True)
        Path('runs').mkdir(exist_ok=True)
        log=open(f'runs/server_{port}.log','ab')
        # --wait keeps an actual WSL client alive. A detached systemd service
        # alone does not prevent WSL from shutting down while Windows trains.
        process=subprocess.Popen(base+['systemd-run','--unit',unit,'--collect','--quiet',
                                 '--wait','--pipe','--setenv=HOME=/root',*args],
                                 stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        host = subprocess.check_output(base+['hostname','-I'],text=True).split()[0]
    else:
        process = subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        os.environ['ROBOCUP_SERVER_PID']=str(process.pid)
        host = '127.0.0.1'
    try:
        time.sleep(.7)
        if windows:
            subprocess.run(base+['systemctl','is-active','--quiet',unit],check=True)
        elif process.poll() is not None:
            raise RuntimeError('rcssserver terminó durante el arranque')
        yield host,port
    finally:
        if windows:
            subprocess.run(base+['systemctl','stop',unit],capture_output=True)
            process.wait(timeout=10)
            log.close()
        else:
            process.terminate()
            process.wait(timeout=10)
