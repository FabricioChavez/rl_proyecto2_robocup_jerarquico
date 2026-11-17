import numpy as np
import pytest
from robocup_rl.comun.protocolo import sexpr,World
from robocup_rl.comun.acciones import command,N_ACTIONS


def test_world_parses_trainer_and_look_consistently():
    payload='42 ((b) 1 2 0.1 0.2) ((p "Emilia" 1) 0 0 0 0 0 0)'
    a=World(sexpr('(see_global '+payload+')'))
    b=World(sexpr('(ok look '+payload+')'))
    assert a.cycle==b.cycle==42
    np.testing.assert_array_equal(a.ball,b.ball)
    np.testing.assert_array_equal(a.players[('Emilia',1)],b.players[('Emilia',1)])


def test_pbrs_telescopes_even_on_timeout():
    gamma=.99
    potentials=[-2.,-.8,-1.2,0.]  # finite terminal potential forced to zero
    shaping=[gamma*b-a for a,b in zip(potentials,potentials[1:])]
    assert sum(gamma**t*f for t,f in enumerate(shaping))==pytest.approx(-potentials[0])


def test_pbrs_option_discount_matches_primitive_discount():
    gamma=.99
    potentials=np.random.default_rng(0).normal(size=17); potentials[-1]=0
    f=gamma*potentials[1:]-potentials[:-1]
    primitive=sum(gamma**t*r for t,r in enumerate(f))
    options=[sum(gamma**j*r for j,r in enumerate(f[i:i+8])) for i in (0,8)]
    assert options[0]+gamma**8*options[1]==pytest.approx(primitive)


def test_actions_have_legal_primitive_command():
    world=World(sexpr('(see_global 1 ((b) 0 0 0 0) ((p "Emilia" 1) 0 0 0 0 0 0) ((p "Emilia" 2) 10 5 0 0 0 0))'))
    for task in ('persecucion','conduccion','definicion','plana'):
        for action in range(N_ACTIONS[task]):
            result=sexpr(command(task,action,world))
            assert result[0] in ('turn','kick','dash')
            assert -180<=result[-1]<=180
