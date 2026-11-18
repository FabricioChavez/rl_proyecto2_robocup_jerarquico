import contextlib
import os
import platform
import signal
import subprocess


@contextlib.contextmanager
def pause_managed_server(port):
    """Congela física y watchdog durante las etiquetas humanas, sin alterar el estado."""
    if platform.system()=='Windows':
        base=['wsl','-d',os.getenv('ROBOCUP_WSL_DISTRO','Ubuntu'),'-u','root','--',
              'systemctl','kill',f'robocup-rl-{port}','--kill-who=main']
        subprocess.run(base+['--signal=STOP'],check=True)
        try:
            yield
        finally:
            subprocess.run(base+['--signal=CONT'],check=True)
    else:
        # Port-specific PID comes from Linux local_server, never a broad pkill.
        pid=int(os.environ['ROBOCUP_SERVER_PID'])
        os.kill(pid,signal.SIGSTOP)
        try:
            yield
        finally:
            os.kill(pid,signal.SIGCONT)
