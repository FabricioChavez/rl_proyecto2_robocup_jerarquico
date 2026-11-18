"""Cliente UDP estricto: no reutiliza observaciones de ciclos anteriores."""
import re
import socket
import time
import numpy as np


def sexpr(text):
    tokens = iter(re.findall(r'"[^"\\]*(?:\\.[^"\\]*)*"|[^\s()]+|[()]', text.strip('\0')))
    def parse(token):
        if token != '(':
            try:
                return float(token)
            except ValueError:
                return token.strip('"')
        result = []
        for item in tokens:
            if item == ')':
                return result
            result.append(parse(item))
        raise ValueError('Mensaje S-expression incompleto')
    return parse(next(tokens))


class UDP:
    def __init__(self, host, port, init):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(5)
        self.addr = (host, port)
        self.send(init)
        # UDP may deliver server_param/player_type before the init reply.
        # Only init confirms the handshake and its dynamically assigned port.
        deadline = time.monotonic() + 5
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('No se recibió la confirmación init')
                self.sock.settimeout(remaining)
                msg, addr = self.sock.recvfrom(65535)
                response = sexpr(msg.decode().strip('\0'))
                if response[0] == 'error':
                    raise RuntimeError(f'Handshake rechazado: {response}')
                if response[0] == 'init':
                    self.initial = response
                    self.addr = addr
                    break
            self.sock.settimeout(5)
        except BaseException:
            self.close()
            raise

    def send(self, command):
        self.sock.sendto((command + '\0').encode(), self.addr)

    def drain(self):
        self.sock.setblocking(False)
        messages = []
        try:
            while True:
                messages.append(self.sock.recv(65535).decode().strip('\0'))
        except BlockingIOError:
            pass
        finally:
            self.sock.settimeout(5)
        return messages

    def receive(self, prefix, after=-1):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            msg = self.sock.recv(65535).decode().strip('\0')
            if msg.startswith(prefix):
                data = sexpr(msg)
                if after < 0 or data[1] > after:
                    return data
            if msg.startswith('(error'):
                raise RuntimeError(msg)
        raise TimeoutError(f'Sin mensaje nuevo {prefix}')

    def close(self):
        self.sock.close()


class World:
    def __init__(self, message):
        if message[:2] == ['ok', 'look']:
            message = ['see_global'] + message[2:]
        self.cycle = int(message[1])
        self.players = {}
        for item in message[2:]:
            if not isinstance(item, list) or not isinstance(item[0], list):
                continue
            ident = item[0]
            if ident[0] == 'b':
                self.ball = np.array(item[1:5], dtype=float)
            elif ident[0] == 'p':
                self.players[(ident[1], int(ident[2]))] = np.array(item[1:7], dtype=float)


class Server:
    """Un servidor exclusivo por entorno. Percepción global privilegiada del trainer."""
    def __init__(self, host='127.0.0.1', port=6000):
        self.attacker = UDP(host, port, '(init Emilia (version 18))')
        self.mate = UDP(host, port, '(init Emilia (version 18))')
        self.goalie = UDP(host, port, '(init Rival (version 18) (goalie))')
        self.defender = UDP(host, port, '(init Rival (version 18))')
        self.clients = [self.attacker, self.mate, self.goalie, self.defender]
        self.trainer = UDP(host, port + 1, '(init (version 18))')
        self.trainer.send('(eye on)')
        self.world = self.look()
        self.stamina = 8000.

    def look(self):
        self.trainer.drain()
        self.trainer.send('(look)')
        return World(self.trainer.receive('(ok look'))

    def move(self, obj, x, y, angle=0, vx=0, vy=0):
        obj = 'ball' if obj == 'b' else obj.replace('p ', 'player ', 1)
        self.trainer.send(f'(move ({obj}) {x} {y} {angle} {vx} {vy})')

    def reset(self, task, rng, scenario):
        self.trainer.send('(change_mode before_kick_off)')
        x = {'persecucion': -10, 'conduccion': 12, 'definicion': 38, 'global': -10, 'plana': -10}[task]
        y = float(rng.uniform(-6, 6))
        self.move('p Emilia 1', x, y, float(rng.uniform(-70,70)) if task in ('persecucion','global','plana') else 0)
        self.move('p Emilia 2', x+7, y-9 if scenario == '2v1' else -30)
        self.move('p Rival 1', 50, float(rng.uniform(-4,4)), 180)
        self.move('p Rival 2', 23 if task != 'definicion' else 32, y+3, 180)
        self.move('b', x+float(rng.uniform(4,10)) if task in ('persecucion','global','plana') else x+0.5, y, 0)
        self.trainer.send('(recover)')
        self.trainer.send('(change_mode play_on)')
        self.stamina = 8000.
        self.world = self.look()
        return self.world

    def tick(self, commands):
        previous = self.world.cycle
        self.trainer.drain()
        for client, command in zip(self.clients, commands):
            client.drain()
            client.send(command)
        for client in self.clients:
            client.send('(done)')
        self.trainer.send('(done)')
        # During referee stoppages the match clock repeats, but the next UDP
        # frame is new. Discarded pre-action packets above must not be reused.
        self.world = World(self.trainer.receive('(see_global'))
        for msg in self.attacker.drain():
            m = re.search(r'\(stamina ([\d.]+)', msg)
            if m:
                self.stamina = float(m[1])
        return self.world

    def close(self):
        for client in self.clients:
            client.send('(bye)')
            client.close()
        self.trainer.send('(bye)')
        self.trainer.close()
