#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``_target_term_in_other_equation`` with the DERIV-CORE extension: a
decorated target (``dv/dx0 * cos``) reserves its derivative core exactly
as a bare ``dv/dx0`` target would -- the lv coupled-junk mechanism, where
the cos-costumed v-target unlocked standalone ``v_t`` for the u-equation's
dominating sum/differentiated identities."""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.evaluators import simple_function_evaluator
from epde.operators.common.right_part_selection import (
    _target_term_in_other_equation)


def _deriv_factor(label):
    return SimpleNamespace(
        structural_label=(label, (1,)),
        is_deriv=True, deriv_code=[0],
        evaluator=SimpleNamespace(_evaluator=simple_function_evaluator))


def _plain_factor(label):
    return SimpleNamespace(
        structural_label=(label, (1,)),
        is_deriv=False, deriv_code=[None, ],
        evaluator=SimpleNamespace(_evaluator=simple_function_evaluator))


def _term(*factors):
    return SimpleNamespace(
        structure=list(factors),
        factors_labels=frozenset(f.structural_label for f in factors))


def _eq_with_target(target_term):
    return SimpleNamespace(target=target_term)


def _eq_other(*terms):
    return SimpleNamespace(
        active_terms_labels=frozenset(t.factors_labels for t in terms))


V_T = _deriv_factor('dv/dx0')
COS = _plain_factor('cos')
U = _plain_factor('u')


class TestTargetCoreBan:

    def test_pure_target_whole_term_leak_still_caught(self):
        tgt = _term(_deriv_factor('dv/dx0'))
        other = _eq_other(_term(_deriv_factor('dv/dx0')), _term(U))
        assert (_target_term_in_other_equation(_eq_with_target(tgt), other)
                == tgt.factors_labels)

    def test_decorated_target_core_leak_caught(self):
        # v-eq target dv/dx0 * cos; u-eq carries standalone dv/dx0 --
        # the lv junk pair's enabling leak.
        tgt = _term(V_T, COS)
        core = _term(_deriv_factor('dv/dx0'))
        other = _eq_other(core, _term(U))
        assert (_target_term_in_other_equation(_eq_with_target(tgt), other)
                == core.factors_labels)

    def test_core_as_factor_of_composite_stays_legal(self):
        # dv/dx0 only inside u * dv/dx0 (a coupling term): NOT a leak --
        # the whole-term-only compromise (the NS v*v_y case).
        tgt = _term(V_T, COS)
        other = _eq_other(_term(U, _deriv_factor('dv/dx0')), _term(U))
        assert _target_term_in_other_equation(
            _eq_with_target(tgt), other) is None

    def test_unrelated_derivative_not_banned(self):
        # d^2v/dx0^2 standalone is NOT the core of dv/dx0 * cos.
        tgt = _term(V_T, COS)
        other = _eq_other(_term(_deriv_factor('d^2v/dx0^2')), _term(U))
        assert _target_term_in_other_equation(
            _eq_with_target(tgt), other) is None

    def test_no_target_returns_none(self):
        other = _eq_other(_term(U))
        assert _target_term_in_other_equation(
            SimpleNamespace(target=None), other) is None


# ---------------------------------------------------------------------------
# The repair a core leak triggers, on a real two-variable system.
# ---------------------------------------------------------------------------

_LIVE_FACTORS = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
DU = 'du/dx0{power: 1.0}'
DV = 'dv/dx0{power: 1.0}'
U_TOKEN = 'u{power: 1.0}'
V_TOKEN = 'v{power: 1.0}'


@pytest.fixture(scope='module')
def uv_pool():
    import epde
    search = epde.EpdeSearch(use_solver=False,
                             verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=10, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory(
        {'u': np.sin(t), 'v': np.cos(t) + 0.3 * np.sin(2 * t)}, domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1)
    return search.pool


def _system(pool, v_target, u_leak=DV):
    """A u-equation carrying ``u_leak`` as a standalone term next to a
    v-equation whose target is ``v_target``. Both are pinned to the fitted
    state the forward RPS pass leaves (targets last, every feature nonzero)."""
    from epde.interface.equation_translator import translate_equation
    system = translate_equation(
        {'u': f'0.5 * {u_leak} + 0.3 * {V_TOKEN} + 0.0 = {DU}',
         'v': f'0.7 * {V_TOKEN} + 0.0 = {v_target}'},
        pool, all_vars=['u', 'v'])
    for var, weights in (('u', [0.5, 0.3, 0.0]), ('v', [0.7, 0.0])):
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


def _soeq_selector():
    """The chromosome-level selector with a per-equation selector that
    records its calls and changes nothing, so the leak repair is the only
    thing acting."""
    from epde.operators.common.right_part_selection import SoEqRightPartSelector
    from epde.operators.utils.template import CompoundOperator

    class RecordingSelector(CompoundOperator):
        key = 'RecordingSelector'

        def __init__(self):
            super().__init__()
            self.calls = []
            self.bans = []

        def apply(self, objective, arguments, banned_sigs=frozenset()):
            self.calls.append(objective.main_var_to_explain)
            self.bans.append(set(banned_sigs))

        def use_default_tags(self):
            self._tags = {'custom level', 'no suboperators', 'inplace'}

    inner = RecordingSelector()
    selector = SoEqRightPartSelector()
    selector.set_suboperators({'eq_right_part_selector': inner})
    return selector, inner


class TestTheCoreLeakRepair:

    def _recorded_repairs(self, pool, monkeypatch, v_target, u_leak=DV):
        import epde.operators.common.right_part_selection as rps
        calls = []

        def recording(equation, shared_sigs, *, preferred_sigs=(),
                      forbidden_sigs=None, max_iter=2000):
            calls.append((equation, set(shared_sigs), set(preferred_sigs),
                          None if forbidden_sigs is None else set(forbidden_sigs)))
            return False

        monkeypatch.setattr(rps, '_break_equation_duplication', recording)
        system = _system(pool, v_target, u_leak)
        selector, _ = _soeq_selector()
        selector.apply(system, {})
        return system, calls

    def test_a_core_leak_also_bans_the_whole_decorated_target(self, uv_pool,
                                                              monkeypatch):
        system, calls = self._recorded_repairs(uv_pool, monkeypatch,
                                               f'{U_TOKEN} * {DV}')
        eq_u, eq_v = system.vals['u'], system.vals['v']
        core_sig = eq_u.structure[0].factors_labels
        banned = {core_sig, eq_v.target.factors_labels}
        assert [(eq, shared, pref) for eq, shared, pref, _ in calls] == [
            (eq_u, banned, {core_sig})]
        # Two equations: what the replacement is judged against is the same set.
        assert calls[0][3] == banned

    def test_a_whole_decorated_target_leak_also_bans_its_core(self, uv_pool,
                                                              monkeypatch):
        system, calls = self._recorded_repairs(uv_pool, monkeypatch,
                                               f'{U_TOKEN} * {DV}',
                                               u_leak=f'{U_TOKEN} * {DV}')
        eq_u, eq_v = system.vals['u'], system.vals['v']
        whole_sig = eq_v.target.factors_labels
        assert eq_u.structure[0].factors_labels == whole_sig
        core_sig = frozenset(f.structural_label for f in eq_v.target.structure
                             if f.variable == 'v')
        assert [(eq, shared, pref) for eq, shared, pref, _ in calls] == [
            (eq_u, {whole_sig, core_sig}, {whole_sig})]

    def test_a_bare_target_leak_bans_just_that_target(self, uv_pool, monkeypatch):
        system, calls = self._recorded_repairs(uv_pool, monkeypatch, DV)
        eq_u, eq_v = system.vals['u'], system.vals['v']
        target_sig = eq_v.target.factors_labels
        assert eq_u.structure[0].factors_labels == target_sig
        assert [(eq, shared, pref) for eq, shared, pref, _ in calls] == [
            (eq_u, {target_sig}, {target_sig})]

    def test_the_core_is_not_wrapped_into_the_decorated_target(self, uv_pool,
                                                               monkeypatch):
        """The wrap's first draw is ``u``, which would turn the leaked
        ``dv/dx0`` into ``u * dv/dx0`` -- the v-equation's whole target, flagged
        again on the very next pass."""
        system = _system(uv_pool, f'{U_TOKEN} * {DV}')
        eq_u, eq_v = system.vals['u'], system.vals['v']
        leak = eq_u.structure[0]
        pool_cls = type(leak.pool)
        stock_create = pool_cls.create
        script = ['u', 'v']
        drawn = []

        def scripted(self, label=None, create_meaningful=False, **kwargs):
            if label is None:
                label = script[min(len(drawn), len(script) - 1)]
                drawn.append(label)
            return stock_create(self, label=label,
                                create_meaningful=create_meaningful, **kwargs)

        monkeypatch.setattr(pool_cls, 'create', scripted)
        selector, inner = _soeq_selector()
        selector.apply(system, {})

        assert drawn == script
        assert any(t is leak for t in eq_u.structure)
        assert leak.factors_labels != eq_v.target.factors_labels
        assert V_TOKEN in leak.name and DV in leak.name and U_TOKEN not in leak.name
        # Forward pass on both equations, then one re-selection of the repaired one.
        assert inner.calls == ['u', 'v', 'u']

    def test_the_whole_target_is_not_randomized_into_its_core(self, uv_pool,
                                                              monkeypatch):
        """The mirror case: the u-equation carries the v-equation's whole target
        ``u * dv/dx0``. At the factor cap the wrap cannot run, so the term is
        randomized -- and a first draw of the bare ``dv/dx0`` would be the
        v-equation's core, flagged on the very next pass."""
        from epde.structure.main_structures import Term
        system = _system(uv_pool, f'{U_TOKEN} * {DV}', u_leak=f'{U_TOKEN} * {DV}')
        eq_u = system.vals['u']
        leak = eq_u.structure[0]
        draws = [['dv/dx0'], ['u']]
        made = []

        def scripted_randomize(self, *args, **kwargs):
            labels = draws[min(len(made), len(draws) - 1)]
            made.append(labels)
            self.structure = [uv_pool.create(label=label)[1] for label in labels]

        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        selector, inner = _soeq_selector()
        selector.apply(system, {})

        assert made == draws
        assert any(t is leak for t in eq_u.structure) and leak.name == U_TOKEN
        assert inner.calls == ['u', 'v', 'u']



class TestTheSharedCompositeTarget:
    """``du/dx0 * dv/dx0`` carries a derivative of each variable, so the
    u- and the v-equation may BOTH pick it as their target. No non-target term
    carries the other equation's target, so the leak repair used to find
    nothing to reroll and left the two equations explaining one quantity."""

    def _pinned(self, pool, texts, weights):
        from epde.interface.equation_translator import translate_equation
        system = translate_equation(texts, pool, all_vars=['u', 'v'])
        for var in ('u', 'v'):
            eq = system.vals[var]
            eq.main_var_to_explain = var
            eq.metaparameters['max_factors_in_term']['value'] = dict(_LIVE_FACTORS)
            for term in eq.structure:
                term.max_factors_in_term = dict(_LIVE_FACTORS)
            w = np.array(weights[var], dtype=float)
            eq.weights_internal = w
            eq.weights_final = w.copy()
            eq.weights_internal_evald = True
            eq.weights_final_evald = True
            eq.target_idx = len(eq.structure) - 1
        return system

    def test_the_later_equation_rerolls_its_target(self, uv_pool, monkeypatch):
        """The reroll must leave the shared composite, and -- since
        re-selection may pick the new term again -- must not make the
        remaining feature ``v`` its component (the scrub would delete it)."""
        from epde.structure.main_structures import Term
        shared = f'{DU} * {DV}'
        system = self._pinned(uv_pool,
                              {'u': f'0.5 * {U_TOKEN} + 0.0 = {shared}',
                               'v': f'0.7 * {V_TOKEN} + 0.0 = {shared}'},
                              {'u': [0.5, 0.0], 'v': [0.7, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        u_target, v_target = eq_u.target, eq_v.target
        assert u_target.factors_labels == v_target.factors_labels
        draws = [('du/dx0', 'dv/dx0'),      # the shared composite itself
                 ('v', 'dv/dx0'),           # would make the feature v its component
                 ('u', 'dv/dx0')]           # acceptable
        made = []

        def scripted_randomize(self, *args, **kwargs):
            made.append((self, draws[min(len(made), len(draws) - 1)]))
            self.structure = [uv_pool.create(label=label)[1] for label in made[-1][1]]

        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        selector, inner = _soeq_selector()
        selector.apply(system, {})

        # Only the v-equation's target was touched, and then re-selected.
        assert [term for term, _ in made] == [v_target] * 3
        assert [labels for _, labels in made] == draws
        assert inner.calls == ['u', 'v', 'v']
        assert eq_u.target is u_target and u_target.name == shared
        assert U_TOKEN in v_target.name and DV in v_target.name

    def test_an_ineligible_decorated_fallback_target_is_rerolled_on_its_own_equation(
            self, uv_pool, monkeypatch):
        """The u-equation's target ``u * dv/dx0`` carries no derivative of u,
        so only RPS's exit fallback installs it; it leaks the v-equation's
        target as its CORE. The reverse scan finds nothing to repair (the
        u-equation carries no standalone ``dv/dx0``), so this pair must reroll
        the u-equation's own target."""
        from epde.structure.main_structures import Term
        system = self._pinned(uv_pool,
                              {'u': f'0.5 * {V_TOKEN} + 0.0 = {U_TOKEN} * {DV}',
                               'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
                              {'u': [0.5, 0.0], 'v': [0.7, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        u_target, v_target = eq_u.target, eq_v.target
        made = []

        def scripted_randomize(self, *args, **kwargs):
            made.append(self)
            self.structure = [uv_pool.create(label=label)[1]
                              for label in ('u', 'du/dx0')]

        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        selector, inner = _soeq_selector()
        selector.apply(system, {})

        assert made == [u_target]
        assert eq_v.target is v_target and v_target.name == DV
        assert inner.calls == ['u', 'v', 'u']

    def test_an_ineligible_fallback_target_is_the_one_rerolled(self, uv_pool,
                                                              monkeypatch):
        """``dv/dx0`` is no target for the u-equation (no derivative of u) --
        only RPS's exit fallback installs one like it. The v-equation's
        legitimate ``dv/dx0`` stays; the u-equation's is rerolled."""
        from epde.structure.main_structures import Term
        system = self._pinned(uv_pool,
                              {'u': f'0.5 * {U_TOKEN} + 0.0 = {DV}',
                               'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'},
                              {'u': [0.5, 0.0], 'v': [0.7, 0.0]})
        eq_u, eq_v = system.vals['u'], system.vals['v']
        u_target, v_target = eq_u.target, eq_v.target
        made = []

        def scripted_randomize(self, *args, **kwargs):
            made.append(self)
            self.structure = [uv_pool.create(label=label)[1] for label in ('v', 'du/dx0')]

        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        selector, inner = _soeq_selector()
        selector.apply(system, {})

        assert made == [u_target]
        assert eq_v.target is v_target and v_target.name == DV
        assert inner.calls == ['u', 'v', 'u']


class TestTheSelectionIsToldWhatOthersReserve:

    def test_forward_pass_and_reselection_bans(self, uv_pool, monkeypatch):
        """Forward pass: each equation hears what the equations selected before
        it reserve. Re-selection after a repair: what every other equation
        reserves -- here the v-equation's whole target and its core."""
        system = _system(uv_pool, f'{U_TOKEN} * {DV}')
        eq_u, eq_v = system.vals['u'], system.vals['v']
        whole = eq_v.target.factors_labels
        core = eq_u.structure[0].factors_labels
        u_reserved = {eq_u.target.factors_labels}
        selector, inner = _soeq_selector()
        selector.apply(system, {})
        assert inner.calls[:3] == ['u', 'v', 'u']
        assert inner.bans[0] == set()
        assert inner.bans[1] == u_reserved
        assert inner.bans[2] == {whole, core}


@pytest.fixture(scope='module')
def uvw_pool():
    """Three variables, so a repair has a THIRD equation's target to avoid."""
    import epde
    search = epde.EpdeSearch(use_solver=False,
                             verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=10, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory(
        {'u': np.sin(t), 'v': np.cos(t) + 0.3 * np.sin(2 * t), 'w': np.sin(3 * t) + 0.5},
        domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1)
    return search.pool


class TestARepairAvoidsEveryOtherEquation:

    def test_a_third_equations_target_is_not_drawn(self, uvw_pool, monkeypatch):
        """The u-equation leaks the v-equation's target. Repairing it must not
        land on the w-equation's target -- that is just the next leak."""
        import epde.operators.common.right_part_selection as rps
        from epde.interface.equation_translator import translate_equation
        from epde.structure.main_structures import Term
        system = translate_equation(
            {'u': f'0.5 * {DV} + 0.0 = {DU}',
             'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}',
             'w': f'0.9 * {V_TOKEN} + 0.0 = dw/dx0{{power: 1.0}}'},
            uvw_pool, all_vars=['u', 'v', 'w'])
        for var, weights in (('u', [0.5, 0.0]), ('v', [0.7, 0.0]), ('w', [0.9, 0.0])):
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
        eq_u, eq_w = system.vals['u'], system.vals['w']
        leak = eq_u.structure[0]
        draws = [['dw/dx0'], ['u']]      # the w-equation's target, then a free draw
        made = []

        def scripted_randomize(self, *args, **kwargs):
            labels = draws[min(len(made), len(draws) - 1)]
            made.append(labels)
            self.structure = [uvw_pool.create(label=label)[1] for label in labels]

        monkeypatch.setattr(rps, '_wrap_term_with_factor', lambda *a, **k: False)
        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        selector, inner = _soeq_selector()
        selector.apply(system, {})

        assert made == draws
        assert leak.factors_labels != eq_w.target.factors_labels
        assert leak.name == U_TOKEN
        # One repair: drawing the w-target would have cost a second one.
        assert inner.calls == ['u', 'v', 'w', 'u']

    def test_the_wrap_avoids_a_third_equations_target(self, uvw_pool, monkeypatch):
        """The u-equation leaks the v-equation's target ``dv/dx0``. Wrapping it
        with ``dw/dx0`` would build exactly the w-equation's target -- the next
        leak, one equation over."""
        from epde.interface.equation_translator import translate_equation
        system = translate_equation(
            {'u': f'0.5 * {DV} + 0.0 = {DU}',
             'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}',
             'w': f'0.9 * {V_TOKEN} + 0.0 = {DV} * dw/dx0{{power: 1.0}}'},
            uvw_pool, all_vars=['u', 'v', 'w'])
        for var, weights in (('u', [0.5, 0.0]), ('v', [0.7, 0.0]), ('w', [0.9, 0.0])):
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
        eq_u, eq_w = system.vals['u'], system.vals['w']
        leak = eq_u.structure[0]
        pool_cls = type(leak.pool)
        stock_create = pool_cls.create
        script = ['dw/dx0', 'v']
        drawn = []

        def scripted(self, label=None, create_meaningful=False, **kwargs):
            if label is not None:
                return stock_create(self, label=label,
                                    create_meaningful=create_meaningful, **kwargs)
            label = script[min(len(drawn), len(script) - 1)]
            drawn.append(label)
            return stock_create(self, label=label,
                                create_meaningful=create_meaningful, **kwargs)

        monkeypatch.setattr(pool_cls, 'create', scripted)
        selector, _ = _soeq_selector()
        selector.apply(system, {})

        assert drawn == script
        assert leak.factors_labels != eq_w.target.factors_labels
        assert V_TOKEN in leak.name and DV in leak.name

    def test_a_banned_feature_the_reductions_leave_alone_does_not_veto_the_reroll(
            self, uvw_pool, monkeypatch):
        """The v-equation shares its target with the u-equation AND carries the
        w-equation's target as a feature. That feature is banned, but the
        reductions never touch it, so it must not veto every reroll draw."""
        from epde.interface.equation_translator import translate_equation
        from epde.structure.main_structures import Term
        shared = f'{DU} * {DV}'
        system = translate_equation(
            {'u': f'0.5 * {U_TOKEN} + 0.0 = {shared}',
             'v': f'0.7 * dw/dx0{{power: 1.0}} + 0.0 = {shared}',
             'w': f'0.9 * {V_TOKEN} + 0.0 = dw/dx0{{power: 1.0}}'},
            uvw_pool, all_vars=['u', 'v', 'w'])
        for var, weights in (('u', [0.5, 0.0]), ('v', [0.7, 0.0]), ('w', [0.9, 0.0])):
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
        eq_v = system.vals['v']
        v_target = eq_v.target
        made = []

        def scripted_randomize(self, *args, **kwargs):
            made.append(self)
            self.structure = [uvw_pool.create(label=label)[1]
                              for label in ('u', 'dv/dx0')]

        monkeypatch.setattr(Term, 'randomize', scripted_randomize)
        selector, _ = _soeq_selector()
        selector.apply(system, {})

        assert made == [v_target]                      # accepted on the first draw
        assert U_TOKEN in v_target.name and DV in v_target.name


def _terms(pool, text):
    """Fresh ``Term`` objects for ``text``, on the live factor budget."""
    from epde.interface.equation_translator import translate_equation
    system = translate_equation(
        {'u': text, 'v': f'0.7 * {V_TOKEN} + 0.0 = {DV}'}, pool, all_vars=['u', 'v'])
    terms = list(system.vals['u'].structure)
    for term in terms:
        term.max_factors_in_term = dict(_LIVE_FACTORS)
    return terms


class TestTheGeneratorsRespectTheBan:
    """``Equation.randomize`` and ``Equation.restore_property`` belong to the
    equation, not to the system, so they draw from the pool with no idea what
    the other equations explain. ``EqRightPartSelector.apply`` scrubs what they
    land on (``_redraw_banned_terms``).

    Measured on Lotka-Volterra before this: 94 of 478 rerolls built another
    equation's target as a standalone term and 61 rode out of the operator,
    leaving the SoEq convergence loop to repair a pass later."""

    @staticmethod
    def _banned(system):
        from epde.operators.common.right_part_selection import _reserved_signatures
        return frozenset(_reserved_signatures(system.vals['v']))

    @staticmethod
    def _stub_fit(monkeypatch, failing):
        """Stand in for the fit/score step of the term sweep: install the unit
        weight layout ``apply`` expects and return a score -- or ``None`` while
        ``failing['now']`` holds, which is how the real operator reports "no
        usable fit" and is what triggers the inf-fitness reroll."""
        from epde.operators.common.right_part_selection import EqRightPartSelector

        def fake_fit(self, objective, subop_args):
            weights = np.ones(len(objective.structure))
            objective.weights_internal = weights
            objective.weights_final = weights.copy()
            objective.weights_internal_evald = True
            objective.weights_final_evald = True
            objective._cached_sw_weights = None
            objective._cached_vc_score = None
            return None if failing['now'] else 0.5

        monkeypatch.setattr(EqRightPartSelector, '_fit_and_score', fake_fit)

    # ---------------------------------------------------------------- helper
    def test_a_freshly_drawn_banned_term_is_redrawn(self, uv_pool):
        """``_regen_or_drop_term`` reads the ban on fresh DRAWS only, so the
        already-banned term in hand has to be pushed past its entry test."""
        from epde.operators.common.right_part_selection import _redraw_banned_terms
        system = _system(uv_pool, DV)
        eq_u = system.vals['u']
        banned = self._banned(system)
        assert eq_u.structure[0].factors_labels in banned

        assert _redraw_banned_terms(eq_u, list(eq_u.structure), banned) is True
        assert not any(t.factors_labels in banned for t in eq_u.structure)
        assert len(eq_u.structure) == 3          # redrawn, not dropped
        assert eq_u.target is not None

    def test_terms_outside_the_ban_are_left_alone(self, uv_pool, monkeypatch):
        from epde.structure.main_structures import Term
        from epde.operators.common.right_part_selection import _redraw_banned_terms
        system = _system(uv_pool, DV, u_leak=U_TOKEN)      # no leak at all
        eq_u = system.vals['u']
        before = [term.name for term in eq_u.structure]
        monkeypatch.setattr(Term, 'randomize',
                            lambda self, *a, **k: pytest.fail('redrew a legal term'))
        assert _redraw_banned_terms(eq_u, list(eq_u.structure),
                                    self._banned(system)) is False
        assert [term.name for term in eq_u.structure] == before

    def test_an_empty_ban_is_a_no_op(self, uv_pool, monkeypatch):
        """A single-equation search passes no ban, and must be untouched."""
        from epde.structure.main_structures import Term
        from epde.operators.common.right_part_selection import _redraw_banned_terms
        eq_u = _system(uv_pool, DV).vals['u']
        before = [term.name for term in eq_u.structure]
        monkeypatch.setattr(Term, 'randomize',
                            lambda self, *a, **k: pytest.fail('redrew without a ban'))
        assert _redraw_banned_terms(eq_u, list(eq_u.structure), frozenset()) is False
        assert [term.name for term in eq_u.structure] == before

    def test_structure_the_equation_arrived_with_is_not_scrubbed(self, uv_pool):
        """Only what a generator JUST drew is handed in. An inherited leak may
        be real coupling, and belongs to the SoEq repair -- which demotes it to
        a FACTOR instead of destroying it."""
        from epde.operators.common.right_part_selection import _redraw_banned_terms
        system = _system(uv_pool, DV)
        eq_u = system.vals['u']
        banned = self._banned(system)
        leak = eq_u.structure[0]
        # the caller hands in every term BUT the inherited one
        assert _redraw_banned_terms(eq_u, eq_u.structure[1:], banned) is False
        assert any(t is leak for t in eq_u.structure)
        assert leak.factors_labels in banned

    # ----------------------------------------------------------- call sites
    def _drive_with_reroll(self, uv_pool, monkeypatch, with_ban):
        """Run the operator with every candidate target scoring ``None`` once,
        so the inf-fitness reroll fires, and script that reroll onto the
        v-equation's target. Returns the u-equation and the ban."""
        from epde.structure.main_structures import Equation
        from epde.operators.common.right_part_selection import EqRightPartSelector
        system = _system(uv_pool, DV)
        eq_u = system.vals['u']
        banned = self._banned(system)
        banned_sigs = banned if with_ban else frozenset()
        failing = {'now': True}
        self._stub_fit(monkeypatch, failing)
        rerolls = []
        real_randomize = Equation.randomize

        def scripted_randomize(self):
            real_randomize(self)
            self.structure = _terms(
                uv_pool, f'0.5 * {DV} + 0.3 * {U_TOKEN} + 0.0 = {DU}')
            self._invalidate_label_cache()
            rerolls.append(None)
            failing['now'] = False        # the reroll "worked"; the sweep scores

        monkeypatch.setattr(Equation, 'randomize', scripted_randomize)
        EqRightPartSelector().apply(objective=eq_u, arguments={},
                                    banned_sigs=banned_sigs)
        assert rerolls, 'the inf-fitness reroll never fired'
        return eq_u, banned

    def test_the_inf_fitness_reroll_does_not_leave_a_leak(self, uv_pool, monkeypatch):
        """Every eligible target scoring ``None`` rerolls the whole equation --
        blind to the system, so the reroll can build the v-equation's target."""
        eq_u, banned = self._drive_with_reroll(uv_pool, monkeypatch, True)
        assert not any(t.factors_labels in banned for t in eq_u.structure), \
            [t.name for t in eq_u.structure]

    def test_without_a_ban_the_same_reroll_keeps_the_term(self, uv_pool, monkeypatch):
        """The control: the scrub is what removes it, not the surrounding loop
        or the simplify pass that follows."""
        eq_u, banned = self._drive_with_reroll(uv_pool, monkeypatch, False)
        assert any(t.factors_labels in banned for t in eq_u.structure)

    def _drive_with_restore_failure(self, uv_pool, monkeypatch, with_ban):
        """Reach the third generator site: ``restore_property`` never manages
        to introduce a derivative of the explained variable, so after
        ``inner_max_iter`` attempts the operator warns and rerolls the whole
        equation. Script that reroll onto the v-equation's target."""
        import warnings as _warnings
        from epde.structure.main_structures import Equation
        from epde.operators.common.right_part_selection import EqRightPartSelector
        system = _system(uv_pool, DV)
        eq_u = system.vals['u']
        banned = self._banned(system)
        # No derivative of u anywhere, so the restore loop must run.
        eq_u.structure = [t for t in eq_u.structure if not t.contains_deriv('u')]
        eq_u._target_term = None
        eq_u._invalidate_label_cache()
        assert eq_u.structure, 'the fixture left nothing to start from'
        self._stub_fit(monkeypatch, {'now': False})
        restores, rerolls = [], []

        def failing_restore(self, **kwargs):
            restores.append(None)        # injects nothing: the property is never met

        def scripted_randomize(self):
            self.structure = _terms(
                uv_pool, f'0.5 * {DV} + 0.3 * {U_TOKEN} + 0.0 = {DU}')
            self._target_term = None
            self._invalidate_label_cache()
            rerolls.append(None)

        monkeypatch.setattr(Equation, 'restore_property', failing_restore)
        monkeypatch.setattr(Equation, 'randomize', scripted_randomize)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            EqRightPartSelector().apply(
                objective=eq_u, arguments={},
                banned_sigs=banned if with_ban else frozenset())
        assert restores, 'the restore loop never ran'
        assert rerolls, 'the give-up reroll never fired'
        assert any('restore_property failed to introduce a deriv' in str(w.message)
                   for w in caught), [str(w.message) for w in caught]
        return eq_u, banned

    def test_the_reroll_after_restore_property_gives_up_does_not_leak(
            self, uv_pool, monkeypatch):
        """The third site. ``restore_property`` cannot supply the derivative,
        so the operator rerolls the equation wholesale -- as blind to the
        system as the inf-fitness reroll, and scrubbed the same way."""
        eq_u, banned = self._drive_with_restore_failure(uv_pool, monkeypatch, True)
        assert not any(t.factors_labels in banned for t in eq_u.structure), \
            [t.name for t in eq_u.structure]

    def test_without_a_ban_the_give_up_reroll_keeps_the_term(self, uv_pool,
                                                             monkeypatch):
        """The control for the test above."""
        eq_u, banned = self._drive_with_restore_failure(uv_pool, monkeypatch, False)
        assert any(t.factors_labels in banned for t in eq_u.structure)

    @staticmethod
    def _no_reroll_scrub(monkeypatch):
        """Silence the reroll scrub (the site-A/B fix) for the exit-guarantee
        tests below.

        The exit fallback is a BACKSTOP: it only runs when the outer loop breaks
        with the target orphaned, which only a reroll does, and the reroll scrub
        now cleans every reserved term out of that structure first -- so with
        the scrub live this state cannot arise (measured: the fallback has never
        fired in 109k live selections). Testing a backstop means removing what
        stands in front of it."""
        import epde.operators.common.right_part_selection as rps
        monkeypatch.setattr(rps, '_redraw_banned_terms', lambda *a, **k: False)

    def _drive_to_the_exit_fallback(self, uv_pool, monkeypatch, structure_text,
                                    with_ban):
        """Reach the exit guarantee: every candidate target scores ``None``, so
        the outer loop rerolls until its cap and breaks with the identity-
        tracked target orphaned by the last reroll. The fallback then has to
        promote one of ``structure_text``'s terms."""
        import warnings as _warnings
        from epde.structure.main_structures import Equation
        from epde.operators.common.right_part_selection import EqRightPartSelector
        system = _system(uv_pool, f'{DU} * {DV}')      # v explains the composite
        eq_u = system.vals['u']
        banned = self._banned(system)
        self._stub_fit(monkeypatch, {'now': True})     # every candidate declined
        self._no_reroll_scrub(monkeypatch)

        def scripted_randomize(self):
            self.structure = _terms(uv_pool, structure_text)
            self._target_term = None
            self._invalidate_label_cache()

        monkeypatch.setattr(Equation, 'randomize', scripted_randomize)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            EqRightPartSelector().apply(
                objective=eq_u, arguments={},
                banned_sigs=banned if with_ban else frozenset())
        assert any('outer loop did not converge' in str(w.message) for w in caught), \
            [str(w.message) for w in caught]
        assert eq_u.target is not None, 'a target must leave RPS'
        return eq_u, banned

    # The reserved composite is FIRST, so plain structure order would take it.
    _EXIT_CHOICE = '0.5 * {shared} + 0.3 * {du} + 0.0 = {u}'

    def test_the_exit_fallback_prefers_an_unreserved_target(self, uv_pool,
                                                            monkeypatch):
        """The exit guarantee installs a target without drawing anything, so it
        is the one place this operator can hand another equation its explained
        quantity. Given a choice, it must not."""
        text = self._EXIT_CHOICE.format(shared=f'{DU} * {DV}', du=DU, u=U_TOKEN)
        eq_u, banned = self._drive_to_the_exit_fallback(uv_pool, monkeypatch,
                                                        text, True)
        assert eq_u.target.factors_labels not in banned, eq_u.target.name
        assert eq_u.target.contains_deriv('u')

    def test_without_a_ban_the_exit_fallback_takes_structure_order(
            self, uv_pool, monkeypatch):
        """The control: the reserved composite comes first, so structure order
        picks it and the ban is what changes the outcome."""
        text = self._EXIT_CHOICE.format(shared=f'{DU} * {DV}', du=DU, u=U_TOKEN)
        eq_u, banned = self._drive_to_the_exit_fallback(uv_pool, monkeypatch,
                                                        text, False)
        assert eq_u.target.factors_labels in banned, eq_u.target.name

    def test_the_exit_fallback_still_installs_a_target_when_all_are_reserved(
            self, uv_pool, monkeypatch):
        """A target MUST leave RPS. When every eligible candidate is reserved
        the ban yields -- the SoEq convergence loop repairs that afterwards,
        whereas an equation with no target breaks ``Equation.evaluate``."""
        text = f'0.5 * {DU} * {DV} + 0.3 * {V_TOKEN} + 0.0 = {U_TOKEN}'
        eq_u, banned = self._drive_to_the_exit_fallback(uv_pool, monkeypatch,
                                                        text, True)
        assert eq_u.target.factors_labels in banned    # yielded, as it must

    def test_a_restore_property_injection_does_not_leave_a_leak(self, uv_pool,
                                                                monkeypatch):
        """``restore_property`` injects a term carrying a derivative of THIS
        equation's variable, so it can only collide with another equation's
        target when that target is a composite carrying one of each -- two
        equations sharing a target. The ban reaches it there too."""
        from epde.structure.main_structures import Equation, Term
        from epde.operators.common.right_part_selection import EqRightPartSelector
        shared = f'{DU} * {DV}'
        system = _system(uv_pool, shared, u_leak=U_TOKEN)
        eq_u = system.vals['u']
        banned = self._banned(system)
        # Strip every derivative of u, so the inner loop must restore one.
        eq_u.structure = [t for t in eq_u.structure if not t.contains_deriv('u')]
        eq_u._target_term = None
        eq_u._invalidate_label_cache()
        self._stub_fit(monkeypatch, {'now': False})
        injected = []

        def scripted_restore(self, **kwargs):
            term = _terms(uv_pool, f'0.5 * {shared} + 0.0 = {DU}')[0]
            injected.append(term)
            self.structure = list(self.structure) + [term]
            self._invalidate_label_cache()

        def scripted_term_randomize(self, *a, **k):
            # a legal derivative of u, so the restore loop's own test is met
            self.structure = [uv_pool.create(label='du/dx0')[1]]

        monkeypatch.setattr(Equation, 'restore_property', scripted_restore)
        monkeypatch.setattr(Term, 'randomize', scripted_term_randomize)
        EqRightPartSelector().apply(objective=eq_u, arguments={}, banned_sigs=banned)

        assert len(injected) == 1, 'restore_property fired %d times' % len(injected)
        assert injected[0].factors_labels not in banned      # redrawn in place
        assert not any(t.factors_labels in banned for t in eq_u.structure), \
            [t.name for t in eq_u.structure]
class TestTheUniquenessPostcondition:
    """``SoEqRightPartSelector.apply`` must not return with an equation still
    carrying another equation's explained quantity.

    Its convergence loop exits either at a fixed point or with its pass budget
    spent. A pass that finds a leak it cannot repair makes no change, so the
    loop reads that as its fixed point and stops -- leaving the leak. Neither
    exit was checked, which is exactly how the generator leaks this module now
    bans went unnoticed."""

    @staticmethod
    def _count_scans(monkeypatch):
        import epde.operators.common.right_part_selection as rps
        stock = rps._target_term_in_other_equation
        calls = []

        def counting(eq_i, eq_j):
            calls.append(None)
            return stock(eq_i, eq_j)

        monkeypatch.setattr(rps, '_target_term_in_other_equation', counting)
        return calls

    def test_a_leak_that_cannot_be_repaired_is_reported(self, uv_pool, monkeypatch):
        """The repair reports no change, so the loop stops at pass 1 believing
        it has converged. The leak is still there and must be said out loud."""
        import warnings as _warnings
        import epde.operators.common.right_part_selection as rps
        system = _system(uv_pool, DV)                  # u carries v's target
        monkeypatch.setattr(rps, '_break_equation_duplication',
                            lambda *a, **k: False)     # nothing can be repaired
        selector, _ = _soeq_selector()
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            selector.apply(system, {})
        messages = [str(w.message) for w in caught]
        assert any('target-term uniqueness not reached' in m for m in messages), messages
        assert any('reserved by v' in m for m in messages), messages

    def test_a_converged_system_says_nothing(self, uv_pool, monkeypatch):
        """The repair works, the next pass is clean, and nothing is reported."""
        import warnings as _warnings
        system = _system(uv_pool, DV)
        selector, _ = _soeq_selector()
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            selector.apply(system, {})
        assert not any('uniqueness not reached' in str(w.message) for w in caught), \
            [str(w.message) for w in caught]
        assert not any(_target_term_in_other_equation(a, b) is not None
                       for a in system.vals for b in system.vals if a is not b)

    def test_a_clean_system_is_not_rescanned(self, uv_pool, monkeypatch):
        """The check costs nothing on the normal path: a pass that finds no
        leak has already proved the invariant, so no second scan is made.
        Two equations = one ordered pair each way = 2 scans, once."""
        import warnings as _warnings
        system = _system(uv_pool, DV, u_leak=U_TOKEN)   # no leak to begin with
        calls = self._count_scans(monkeypatch)
        selector, _ = _soeq_selector()
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            selector.apply(system, {})
        assert len(calls) == 2, len(calls)
        assert not any('uniqueness not reached' in str(w.message) for w in caught)


class TestTheTranslatedSystemKnowsItsVariables:

    def test_each_translated_equation_explains_its_own_variable(self, uvw_pool):
        """translate_equation's dict overload used to rebind its loop key in the
        sparsity loop, so every equation came back explaining all_vars[-1]
        ('w'); SolverBasedFitness then observed w for all three."""
        from epde.interface.equation_translator import translate_equation
        system = translate_equation(
            {'u': f'0.5 * {V_TOKEN} + 0.0 = {DU}',
             'v': f'0.7 * {U_TOKEN} + 0.0 = {DV}',
             'w': f'0.2 * {U_TOKEN} + 0.0 = dw/dx0{{power: 1.0}}'},
            uvw_pool, all_vars=['u', 'v', 'w'])
        assert {var: system.vals[var].main_var_to_explain
                for var in system.vars_to_describe} == {'u': 'u', 'v': 'v', 'w': 'w'}
