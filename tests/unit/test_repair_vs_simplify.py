#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The SoEq leak repair must not build a term that ``simplify_equation``'s own
algebra turns straight back into the leak.

A leaked standalone ``dv/dx0`` in the u-equation is wrapped into a composite
term. Under a ``u*du/dx0`` target the wrap ``u*dv/dx0`` shares ``u`` with the
target, and once the refit leaves the two alone, common-factor cancellation
strips ``u`` again; under a ``(du/dx0)^2`` target the power merge
``(dv/dx0)^2`` is undone by degree reduction. Both reductions are correct --
the wrapped law is the leaked law with a factor multiplied through -- so the
repair is what has to avoid them (``_undone_by_simplify``) -- against the
CURRENT target, on both sides of the pair."""

import numpy as np
import pytest

from epde.operators.common.right_part_selection import (
    _break_equation_duplication, _symbolic_reductions, _wrap_term_with_factor)

_LIVE_FACTORS = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
DU = 'du/dx0{power: 1.0}'
DV = 'dv/dx0{power: 1.0}'
U = 'u{power: 1.0}'
V = 'v{power: 1.0}'


@pytest.fixture(scope='module')
def uv_pool():
    """Two variables with squared data and derivative tokens, to second order."""
    import epde
    search = epde.EpdeSearch(use_solver=False,
                             verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=10, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory(
        {'u': np.sin(t) + 1.5, 'v': np.cos(t) + 0.3 * np.sin(2 * t) + 1.5},
        domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=2,
                       deriv_fun_pow=2)
    return search.pool


def _system(pool, u_text, u_weights, v_text='0.7 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}',
            v_weights=(0.7, 0.0)):
    """Both equations pinned to the fitted state RPS leaves: target last,
    every feature nonzero, the live two-factor budget."""
    from epde.interface.equation_translator import translate_equation
    system = translate_equation({'u': u_text, 'v': v_text}, pool, all_vars=['u', 'v'])
    for var, weights in (('u', u_weights), ('v', v_weights)):
        eq = system.vals[var]
        eq.main_var_to_explain = var
        eq.metaparameters['max_factors_in_term']['value'] = dict(_LIVE_FACTORS)
        for term in eq.structure:
            term.max_factors_in_term = dict(_LIVE_FACTORS)
        w = np.array(weights, dtype=float)
        eq.weights_internal = w
        eq.weights_final = w.copy()
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        eq.target_idx = len(eq.structure) - 1
    return system


def _terms(pool, *texts):
    """Real Term objects for ``texts``, via one translated u-equation whose
    target is the last text."""
    body = ' + '.join(f'1.0 * {text}' for text in texts[:-1])
    system = _system(pool, f'{body} + 0.0 = {texts[-1]}', [1.0] * (len(texts) - 1) + [0.0])
    return system.vals['u'].structure


class TestTheSymbolicReductions:

    @pytest.mark.parametrize('term, other, reduced', [
        (f'{U} * {DV}', f'{U} * {DU}', DV),                         # cancellation
        ('dv/dx0{power: 2.0}', 'du/dx0{power: 2.0}', DV),            # degree reduction
        (f'u{{power: 2.0}} * {DV}', f'{U} * {DU}', f'{U} * {DV}'),  # partial cancellation
        (f'{V} * {DV}', DU, f'{V} * {DV}'),                          # nothing shared
        (f'{U} * {DV}', DU, f'{U} * {DV}'),                          # bare target: kept
    ], ids=['cancellation', 'degree', 'partial', 'unrelated', 'bare-target'])
    def test_matches_what_simplify_would_leave(self, uv_pool, term, other, reduced):
        term_, other_ = _terms(uv_pool, term, other)
        expected = _terms(uv_pool, reduced, DU)[0].factors_labels
        before = term_.factors_labels
        assert _symbolic_reductions(term_, other_)[0] == expected
        assert term_.factors_labels == before, 'the reduction must work on clones'

    def test_a_term_reduced_to_nothing_is_none(self, uv_pool):
        term, other = _terms(uv_pool, U, f'{U} * {DU}')
        assert _symbolic_reductions(term, other) == (None, frozenset(
            _terms(uv_pool, DU, DU)[0].factors_labels))


def _unit_power(factor):
    idx = next(i for i, d in factor.params_description.items() if d['name'] == 'power')
    factor.set_param(1, idx=idx)
    return factor


def _scripted_pool_draws(monkeypatch, pool_cls, script):
    """The wrap's anonymous draws follow ``script``, each at power 1 (the pool
    would otherwise pick the power at random)."""
    stock_create = pool_cls.create
    drawn = []

    def scripted(self, label=None, create_meaningful=False, **kwargs):
        if label is not None:
            return stock_create(self, label=label,
                                create_meaningful=create_meaningful, **kwargs)
        label = script[min(len(drawn), len(script) - 1)]
        drawn.append(label)
        key, factor = stock_create(self, label=label,
                                   create_meaningful=create_meaningful, **kwargs)
        return key, _unit_power(factor)

    monkeypatch.setattr(pool_cls, 'create', scripted)
    return drawn


class TestTheWrapAvoidsWhatSimplifyUndoes:

    def test_a_factor_the_target_shares_is_skipped(self, uv_pool, monkeypatch):
        """``u*dv/dx0`` under ``u*du/dx0`` cancels back to ``dv/dx0``."""
        system = _system(uv_pool, f'0.5 * {DV} + 0.0 = {U} * {DU}', [0.5, 0.0])
        eq_u = system.vals['u']
        leak = eq_u.structure[0]
        banned = {leak.factors_labels}
        drawn = _scripted_pool_draws(monkeypatch, type(leak.pool), ['u', 'v'])
        np.random.seed(0)
        assert _wrap_term_with_factor(eq_u, leak, banned) is True
        assert drawn == ['u', 'v']
        assert 'v{power' in leak.name and 'u{power' not in leak.name
        assert not set(_symbolic_reductions(leak, eq_u.target)) & banned

    def test_a_power_merge_under_an_even_target_is_skipped(self, uv_pool, monkeypatch):
        """``dv/dx0`` wrapped with itself is ``(dv/dx0)^2``; under the target
        ``(du/dx0)^2`` degree reduction halves it back to ``dv/dx0``."""
        system = _system(uv_pool, f'0.5 * {DV} + 0.0 = du/dx0{{power: 2.0}}', [0.5, 0.0])
        eq_u = system.vals['u']
        leak = eq_u.structure[0]
        banned = {leak.factors_labels}
        drawn = _scripted_pool_draws(monkeypatch, type(leak.pool), ['dv/dx0', 'v'])
        np.random.seed(0)
        assert _wrap_term_with_factor(eq_u, leak, banned) is True
        assert drawn == ['dv/dx0', 'v']
        assert 'v{power' in leak.name
        assert not set(_symbolic_reductions(leak, eq_u.target)) & banned

    def test_only_the_current_target_is_checked(self, uv_pool, monkeypatch):
        """``u*dv/dx0`` would cancel back to the leak only if the next sweep
        picked the candidate ``u*du/dx0`` AND the refit zeroed everything
        else. The current target ``d^2u/dx0^2`` shares nothing, so the wrap
        stands -- the double pendulum's ``cos(D)*th2''`` next to a
        ``cos(D)*th1'`` term is this case, and there the real refit never
        cancels it."""
        system = _system(uv_pool,
                         f'0.5 * {DV} + 0.2 * {U} * {DU} + 0.0 = d^2u/dx0^2{{power: 1.0}}',
                         [0.5, 0.2, 0.0])
        eq_u = system.vals['u']
        leak = eq_u.structure[0]
        drawn = _scripted_pool_draws(monkeypatch, type(leak.pool), ['u', 'v'])
        np.random.seed(0)
        assert _wrap_term_with_factor(eq_u, leak, {leak.factors_labels}) is True
        assert drawn == ['u']
        assert 'u{power' in leak.name and 'dv/dx0' in leak.name

    def test_a_coupling_factor_the_target_lacks_is_kept(self, uv_pool, monkeypatch):
        """The wrap's reason to exist: the leaked quantity survives as a factor
        of a coupling term (``u*dv/dx0`` under a bare ``du/dx0``)."""
        system = _system(uv_pool, f'0.5 * {DV} + 0.0 = {DU}', [0.5, 0.0])
        eq_u = system.vals['u']
        leak = eq_u.structure[0]
        drawn = _scripted_pool_draws(monkeypatch, type(leak.pool), ['u'])
        np.random.seed(0)
        assert _wrap_term_with_factor(eq_u, leak, {leak.factors_labels}) is True
        assert drawn == ['u']
        assert 'u{power' in leak.name and 'dv/dx0' in leak.name


class TestTheRandomizeFallbackAvoidsWhatSimplifyUndoes:

    def test_a_draw_that_cancels_back_to_the_leak_is_skipped(self, uv_pool, monkeypatch):
        import epde.operators.common.right_part_selection as rps
        from epde.structure.main_structures import Term
        system = _system(uv_pool, f'0.5 * {DV} + 0.0 = {U} * {DU}', [0.5, 0.0])
        eq_u = system.vals['u']
        leak = eq_u.structure[0]
        leak_sig = leak.factors_labels
        draws = [['u', 'dv/dx0'], ['v']]
        made = []

        def scripted_randomize(self, *args, **kwargs):
            labels = draws[min(len(made), len(draws) - 1)]
            made.append(labels)
            self.structure = [_unit_power(uv_pool.create(label=label)[1])
                              for label in labels]

        monkeypatch.setattr(rps, '_wrap_term_with_factor', lambda *a, **k: False)
        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        assert _break_equation_duplication(eq_u, {leak_sig},
                                           preferred_sigs={leak_sig}) is True
        assert made == draws
        assert any(t is leak for t in eq_u.structure) and leak.name == V

    def test_a_draw_that_cancels_the_target_down_to_the_leak_is_skipped(
            self, uv_pool, monkeypatch):
        """The other side of the pair: ``v*du/dx0`` under ``du/dx0*dv/dx0``
        shares ``du/dx0``, and cancelling it leaves the TARGET as ``dv/dx0``
        -- the leak, now where no repair can reach it."""
        import epde.operators.common.right_part_selection as rps
        from epde.structure.main_structures import Term
        system = _system(uv_pool, f'0.5 * {DV} + 0.0 = {DU} * {DV}', [0.5, 0.0])
        eq_u = system.vals['u']
        leak = eq_u.structure[0]
        leak_sig = leak.factors_labels
        draws = [['v', 'du/dx0'], ['v']]
        made = []

        def scripted_randomize(self, *args, **kwargs):
            labels = draws[min(len(made), len(draws) - 1)]
            made.append(labels)
            self.structure = [_unit_power(uv_pool.create(label=label)[1])
                              for label in labels]

        monkeypatch.setattr(rps, '_wrap_term_with_factor', lambda *a, **k: False)
        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        assert _break_equation_duplication(eq_u, {leak_sig},
                                           preferred_sigs={leak_sig}) is True
        assert made == draws
        assert leak.name == V


class TestARerolledTargetIsJudgedAsTheNextTarget:

    def test_a_target_that_cancels_a_feature_onto_the_shared_composite_is_skipped(
            self, uv_pool, monkeypatch):
        """v-equation ``0.7 * u*dv/dx0 = du/dx0*dv/dx0`` shares its target with
        the u-equation. Rerolling it into ``du/dx0 * (dv/dx0)^2`` would, once
        re-selected, cancel ``dv/dx0`` against the feature ``u*dv/dx0`` and
        leave ``du/dx0*dv/dx0`` again."""
        from epde.structure.main_structures import Term
        shared = f'{DU} * {DV}'
        system = _system(uv_pool, f'0.5 * {U} + 0.0 = {shared}', [0.5, 0.0],
                         v_text=f'0.7 * {U} * {DV} + 0.0 = {shared}')
        eq_v = system.vals['v']
        target = eq_v.target
        shared_sig = target.factors_labels
        draws = [[('du/dx0', 1), ('dv/dx0', 2)], [('u', 1), ('du/dx0', 1)]]
        made = []

        def scripted_randomize(self, *args, **kwargs):
            labels = draws[min(len(made), len(draws) - 1)]
            made.append(labels)
            structure = []
            for label, power in labels:
                factor = _unit_power(uv_pool.create(label=label)[1])
                idx = next(i for i, d in factor.params_description.items()
                           if d['name'] == 'power')
                factor.set_param(power, idx=idx)
                structure.append(factor)
            self.structure = structure

        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        assert _break_equation_duplication(eq_v, {shared_sig},
                                           preferred_sigs={shared_sig}) is True
        assert made == draws
        assert any(t is target for t in eq_v.structure)
        assert U in target.name and DU in target.name


class TestTheRerollNeverLeavesADuplicate:

    def test_a_cap_hit_on_a_two_term_equation_redraws_for_uniqueness(
            self, uv_pool, monkeypatch):
        """The post-loop drop refuses to go below two terms, so a duplicate left
        by a cap-hit would ride out of the repair -- and the next
        ``EqRightPartSelector.apply`` asserts on duplicates."""
        import epde.operators.common.right_part_selection as rps
        from epde.structure.main_structures import Term
        system = _system(uv_pool, f'0.5 * {DV} + 0.0 = {DU}', [0.5, 0.0])
        eq_u = system.vals['u']
        leak = eq_u.structure[0]
        made = []

        def scripted_randomize(self, *args, **kwargs):
            made.append(None)
            # Duplicate the target until the loop gives up, then a free term.
            label = 'du/dx0' if len(made) <= 5 else 'v'
            self.structure = [_unit_power(uv_pool.create(label=label)[1])]

        monkeypatch.setattr(rps, '_wrap_term_with_factor', lambda *a, **k: False)
        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        assert _break_equation_duplication(eq_u, {leak.factors_labels},
                                           preferred_sigs={leak.factors_labels},
                                           max_iter=5) is True
        assert len(made) == 6                      # 5 rejected, then the redraw
        assert leak.name == V
        signatures = {t.factors_labels for t in eq_u.structure}
        assert len(signatures) == len(eq_u.structure)
