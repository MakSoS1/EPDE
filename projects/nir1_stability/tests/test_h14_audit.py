"""Strict H14 audit cannot cherry-pick only successful synthetic systems."""
import pytest

from projects.nir1_stability.nir1.analyze_h14 import (
    CASES, exact_two_sided_signflip, analyze,
)


def test_exact_system_level_signflip_no_pseudoreplicate_seed_counting():
    assert len(CASES)==11 and len(set(CASES))==11
    assert exact_two_sided_signflip([0,0,0])==1.0
    assert exact_two_sided_signflip([1])==1.0
    assert exact_two_sided_signflip([1,1])==.5
    assert exact_two_sided_signflip([1]*6)==.03125
    assert exact_two_sided_signflip([1,1,-1])==1.0


def test_scientific_summary_requires_every_predeclared_condition():
    with pytest.raises(ValueError,match="INCOMPLETE"):
        analyze([],research_sha="0123456789abcdef")


def test_no_abstention_can_be_coded_as_correct():
    from projects.nir1_stability.nir1.closed_evolution_selector import (
        choose_causal_closed_front,
    )
    false="1.0 * du/dx0{power: 1.0} * u{power: 1.0} = d^2u/dx0^2{power: 1.0}"
    assert choose_causal_closed_front([[false]],[[0.,0.]]) is None
