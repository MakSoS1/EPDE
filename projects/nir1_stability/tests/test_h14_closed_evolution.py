"""Physics-prior H14 does NOT see simulator truth or alter search budget."""
import pytest
from projects.nir1_stability.nir1.closed_evolution_selector import (
    is_closed_evolution_equation, choose_causal_closed_front,
)


FIRST="-0.57 * du/dx1{power: 1.0} + -0.24 * u{power: 1.0} = du/dx0{power: 1.0}"
SECOND="0.49 * d^2u/dx1^2{power: 1.0} + -0.1225 * u{power: 1.0} = d^2u/dx0^2{power: 1.0}"
FALSE="-0.92 * du/dx0{power: 1.0} * u{power: 3.0} = d^2u/dx0^2{power: 1.0}"
DAMPED="0.49 * d^2u/dx1^2{power: 1.0} + -0.25 * du/dx0{power: 1.0} = d^2u/dx0^2{power: 1.0}"


def test_explicit_single_evolution_forms():
    assert is_closed_evolution_equation(FIRST)
    assert is_closed_evolution_equation(SECOND)
    assert not is_closed_evolution_equation(FALSE)
    assert not is_closed_evolution_equation(DAMPED)
    assert not is_closed_evolution_equation("u = u")


def test_truth_free_selection_discards_minimum_implicit_fitness():
    front=[[FALSE],[FIRST],[SECOND]]
    obj=[[.001,.0],[.07,.0],[.09,.4]]
    assert choose_causal_closed_front(front,obj)==1
    assert choose_causal_closed_front(front,obj,truth_labels=["untrusted"])==1
    assert choose_causal_closed_front([[FALSE]],[[.01,0]]) is None


def test_multiequation_must_be_explicit_in_every_component():
    assert choose_causal_closed_front([[FIRST,FALSE],[FIRST,SECOND]],
                                      [[.01,.0,.01,.0],[.04,.1,.05,.1]])==1
    with pytest.raises(ValueError):
        choose_causal_closed_front([[FIRST]],[])


def test_rejects_nan_and_nonfinite_objective():
    assert choose_causal_closed_front([[FIRST],[SECOND]],
                                      [[float("nan"),0],[.1,0]])==1
