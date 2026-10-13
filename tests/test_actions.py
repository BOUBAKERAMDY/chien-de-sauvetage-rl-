import numpy as np
import pytest

from rescue_dog.visualization.actions import Action, ActionController, parse_actions


def test_parse_actions():
    acts = parse_actions("avancer 1; tourner -90 ; reculer 0.5; stop")
    assert [(a.kind, a.value) for a in acts] == [
        ("avancer", 1.0), ("tourner", -90.0), ("reculer", 0.5), ("stop", 0.0)]
    with pytest.raises(ValueError):
        parse_actions("sauter 1")


def test_forward_stops_at_distance():
    c = ActionController(verbose=False)
    c.add(Action("avancer", 1.0))
    v, _, _ = c.command(np.zeros(2), 0.0, 0.02)
    assert v > 0
    c.command(np.array([1.01, 0.0]), 0.0, 0.02)  # cible dépassée -> fini
    assert c.idle
    assert c.command(np.array([1.01, 0.0]), 0.0, 0.02) == (0.0, 0.0, 0.0)


def test_turn_direction_and_completion():
    c = ActionController(verbose=False)
    c.add(Action("tourner", 90))
    _, w, _ = c.command(np.zeros(2), 0.0, 0.02)
    assert w > 0  # gauche
    c.command(np.zeros(2), np.radians(89), 0.02)
    assert c.idle


def test_stop_clears_queue():
    c = ActionController(verbose=False)
    c.add_many(parse_actions("avancer 1; tourner 90"))
    c.command(np.zeros(2), 0.0, 0.02)
    c.add(Action("stop"))
    assert c.idle
