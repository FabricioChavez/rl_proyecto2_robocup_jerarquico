"""Double+Dueling DQN y MP-DQN propios: replay, gradientes y checkpoints PTH."""
from copy import deepcopy
import random
import numpy as np
import torch
from torch.nn import functional as F
from modelo.arquitecturas.conduccion import DuelingQ, Parametros, MultiPassQ
from modelo.pesos import dispositivo_elegido
from entorno.conduccion import DISCRETAS, ESTADOS_CONDUCCION, ConduccionEnv


class Replay:
    def __init__(self, capacidad, estados=ESTADOS_CONDUCCION):
        self.capacidad = capacidad
        self.pos = self.n = 0
        self.data = dict(s=np.empty((capacidad, estados), np.float32),
                         sn=np.empty((capacidad, estados), np.float32),
                         a=np.empty(capacidad, np.int64),
                         p=np.empty((capacidad, 4), np.float32),
                         r=np.empty(capacidad, np.float32),
                         done=np.empty(capacidad, np.float32))

    def add(self, s, a, p, r, sn, done):
        for key, value in dict(s=s, a=a, p=p, r=r, sn=sn, done=done).items():
            self.data[key][self.pos] = value
        self.pos = (self.pos+1) % self.capacidad
        self.n = min(self.n+1, self.capacidad)

    def sample(self, batch, device):
        indices = np.random.randint(self.n, size=batch)
        return {key: torch.as_tensor(value[indices], device=device)
                for key, value in self.data.items()}


def mascara_legal(obs, algoritmo):
    # Restricción física compartida: no elegir kick si no puede patear.
    count = 8 if algoritmo == 'ddqn_dueling' else 3
    legal = torch.ones((len(obs), count), dtype=torch.bool, device=obs.device)
    kickable = obs[:, 0]*105 <= 1.085
    if count == 8:
        legal[:, 4:] = kickable[:, None]
    else:
        legal[:, 2] = kickable
    return legal


class AgenteConduccion:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(dispositivo_elegido(args.dispositivo))
        random.seed(args.semilla)
        np.random.seed(args.semilla)
        torch.manual_seed(args.semilla)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(args.semilla)
        self.pasos = self.updates = self.episodios = 0
        self.es_discreto = args.algoritmo == 'ddqn_dueling'
        self.q = (DuelingQ(ESTADOS_CONDUCCION, args.red) if self.es_discreto else MultiPassQ(ESTADOS_CONDUCCION, args.red)).to(self.device)
        self.q_target = deepcopy(self.q).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.actor = self.actor_target = self.actor_optimizer = None
        if not self.es_discreto:
            self.actor = Parametros(ESTADOS_CONDUCCION, args.red).to(self.device)
            self.actor_target = deepcopy(self.actor).requires_grad_(False)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.replay = Replay(args.buffer)
        self.curriculum = dict(indice=0, ventana=[], episodios_etapa=0)

    def epsilon(self):
        fraction = min(1., self.pasos/self.args.epsilon_pasos)
        return self.args.epsilon_inicial+fraction*(self.args.epsilon_final-self.args.epsilon_inicial)

    @torch.no_grad()
    def actuar(self, obs, explorar=True):
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        legal = mascara_legal(s, self.args.algoritmo)
        random_action = explorar and np.random.random() < self.epsilon()
        if self.es_discreto:
            a = (int(np.random.choice(np.flatnonzero(legal[0].cpu().numpy()))) if random_action else
                 int(self.q(s).masked_fill(~legal, -torch.inf).argmax(1).item()))
            tipo, params = DISCRETAS[a]
            return a, np.asarray(params, np.float32), (tipo, params)
        params = (torch.rand((1, 4), device=self.device)*2-1 if random_action else
                  self.actor(s).clamp(-1, 1))
        a = (int(np.random.choice(np.flatnonzero(legal[0].cpu().numpy()))) if random_action else
             int(self.q(s, params).masked_fill(~legal, -torch.inf).argmax(1).item()))
        p = params[0].cpu().numpy()
        return a, p, (a, p)

    def actualizar(self):
        batch = self.replay.sample(self.args.batch, self.device)
        s, sn = batch['s'], batch['sn']
        with torch.no_grad():
            legal = mascara_legal(sn, self.args.algoritmo)
            if self.es_discreto:
                # Double: selección online, evaluación con target.
                selected = self.q(sn).masked_fill(~legal, -torch.inf).argmax(1)
                future = self.q_target(sn).gather(1, selected[:, None]).squeeze(1)
            else:
                next_params = self.actor_target(sn).clamp(-1, 1)
                future = self.q_target(sn, next_params).masked_fill(~legal, -torch.inf).max(1).values
            target = batch['r']+self.args.gamma*(1-batch['done'])*future
        values = self.q(s) if self.es_discreto else self.q(s, batch['p'])
        predicted = values.gather(1, batch['a'][:, None]).squeeze(1)
        q_loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.args.norma_gradiente)
        self.optimizer.step()
        actor_loss = 0.
        if not self.es_discreto:
            raw = self.actor(s)
            parameters = raw.detach().clamp(-1, 1).requires_grad_(True)
            legal = mascara_legal(s, self.args.algoritmo)
            objective = self.q(s, parameters).masked_fill(~legal, 0.).sum(1).mean()
            gradient, = torch.autograd.grad(objective, parameters)
            # Inverting gradients: el signo se invierte fuera del intervalo.
            scale = torch.where(gradient > 0, (1-raw.detach())/2, (raw.detach()+1)/2)
            self.actor_optimizer.zero_grad(set_to_none=True)
            raw.backward(-gradient*scale)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.norma_gradiente)
            self.actor_optimizer.step()
            actor_loss = -float(objective.detach())
        with torch.no_grad():
            for target_net, online in ((self.q_target, self.q), (self.actor_target, self.actor)):
                if online is not None:
                    for target_param, param in zip(target_net.parameters(), online.parameters()):
                        target_param.lerp_(param, self.args.tau)
        self.updates += 1
        return dict(perdida_q=float(q_loss.detach()), perdida_actor=actor_loss,
                    q_media=float(predicted.detach().mean()), objetivo_td=float(target.mean()))

    def guardar(self, path):
        rng = np.random.get_state()
        modules = {key: (getattr(self, key).state_dict() if getattr(self, key) is not None else None)
                   for key in ('q', 'q_target', 'actor', 'actor_target', 'optimizer', 'actor_optimizer')}
        torch.save(dict(formato=ConduccionEnv.VERSION, generador='torch_propio',
                        argumentos=vars(self.args), pasos=self.pasos, updates=self.updates,
                        curriculum=self.curriculum,
                        episodios=self.episodios, modules=modules,
                        replay={key: torch.from_numpy(value[:self.replay.n].copy())
                                for key, value in self.replay.data.items()},
                        replay_pos=self.replay.pos, replay_n=self.replay.n,
                        rng_python=random.getstate(), rng_numpy=(rng[0], rng[1].tolist(), *rng[2:]),
                        rng_torch=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else []), path)

    def restaurar(self, data, entrenar=True):
        if data.get('formato') != ConduccionEnv.VERSION or data.get('generador') != 'torch_propio':
            raise ValueError('Checkpoint de reglas anteriores: usa un --lote nuevo para recuperación')
        for key in ('algoritmo', 'red', 'metros', 'radio_control', 'distancia_maxima',
                    'gracia_control', 'gamma', 'premio', 'perdida', 'intercepcion', 'costo',
                    'pbrs', 'escala_pbrs', 'horizonte', 'arquero', 'gracia_recuperacion',
                    'distancia_recuperacion', 'posesion_rival_maxima', 'costo_sin_control',
                    'modo_curriculo', 'umbral_curriculo', 'ventana_curriculo',
                    'min_episodios_curriculo', 'curriculo'):
            if data['argumentos'][key] != getattr(self.args, key):
                raise ValueError(f'El argumento {key} no coincide con el checkpoint')
        for key, state in data['modules'].items():
            if state is not None and (entrenar or 'optimizer' not in key):
                getattr(self, key).load_state_dict(state)
        self.pasos, self.updates, self.episodios = data['pasos'], data['updates'], data['episodios']
        self.curriculum = deepcopy(data['curriculum'])
        if entrenar:
            if data['replay_n'] > self.replay.capacidad:
                raise ValueError('--buffer es menor que el replay guardado')
            for key, value in data['replay'].items():
                self.replay.data[key][:len(value)] = value.cpu().numpy()
            self.replay.n = data['replay_n']
            self.replay.pos = data['replay_pos'] if self.replay.n == self.replay.capacidad else self.replay.n
            random.setstate(data['rng_python'])
            tag, keys, pos, gauss, cached = data['rng_numpy']
            np.random.set_state((tag, np.asarray(keys, np.uint32), pos, gauss, cached))
            torch.set_rng_state(data['rng_torch'].cpu())
            if self.device.type == 'cuda' and len(data['rng_cuda']) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([t.cpu() for t in data['rng_cuda']])
            for group in self.optimizer.param_groups:
                group['lr'] = self.args.lr
            if self.actor_optimizer:
                for group in self.actor_optimizer.param_groups:
                    group['lr'] = self.args.lr_actor
