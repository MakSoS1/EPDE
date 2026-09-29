#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:50:55 2021

@author: mike_ubuntu
"""
import time
from typing import List, Dict, Tuple, Any

import numpy as np
from copy import deepcopy
import warnings

from functools import partial
from collections import defaultdict

import epde.globals as global_var
from epde.interface.search_config import active_config
from epde.supplementary import filter_powers

from epde.operators.utils.template import CompoundOperator, dictApplyUFunc, dictAdd
from epde.decorators import HistoryExtender
from epde.structure.main_structures import Term, Equation
from epde.evaluators import simple_function_evaluator

from epde.operators.common.stability import (GramSetup, VaryingCoefSetup)

from epde import _loop_stats

# One-shot guard so a permanently-broken super-Gram fast path is reported
# once instead of being silently swallowed on every term sweep.
_SUPER_GRAM_FAILURE_REPORTED = False


def loDs_to_DoLs(arg: List[Dict[int, Any]]) -> Dict[int, list]:
    res = defaultdict(list)
    for feature in arg:
        for samp_key, samp_item in feature.items():
            res[samp_key].append(samp_item)
    return res

class EqRightPartSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    THIS OPERATOR OWNS THE FIT. Each candidate target is sparsified, given
    coefficients and only then scored (``_fit_and_score``) -- the fitness
    hosts are pure scorers that neither fit nor prune. It is also the single
    place that prunes: the one ``remove_zero_terms`` at the end of ``apply``,
    after the sweep has chosen a winner. (That prune cannot move into the
    sparsity operator, which runs per candidate and must keep every
    candidate term.)

    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator.
        Suboperators, performing tasks of equation processing. Three are
        present, applied in this order per candidate target: ``sparsity``
        (support selection), ``coeff_calc`` (physical magnitudes) and
        ``fitness_calculation`` (the discrepancy candidates are ranked by).

    Methods:
    -----------
    apply(equation)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``

    
    '''
    key = 'FitnessCheckingRightPartSelector'

    @staticmethod
    @_loop_stats.timed('EqRPS.gram_super')
    def _precompute_super_gram(objective: Equation) -> None:
        """Build a per-equation super-Gram over all structure terms and
        attach it to ``objective._gram_super`` for the upcoming term-sweep.

        Each candidate target_idx in the sweep then derives its
        ``GramSetup`` view via :meth:`GramSetup.from_full` (pure slicing,
        no recompute). On any failure (e.g. non-finite term evaluations,
        non-grid-shaped weights) clear the slot and let downstream
        ``VWSRSparsity.apply`` fall back to its legacy per-target path.
        """
        try:
            # Guard the source this function actually reads: the per-trajectory
            # manager (the old ``grid_cache`` check survived the multisample
            # port but no longer gates anything used below).
            if getattr(global_var, 'samples_manager', None) is None:
                objective._gram_super = None
                return
            sample_weights = global_var.samples_manager.gFunc('dm')
            grid_shapes = global_var.samples_manager.inner_shapes
            feats = [term.evaluate() # grids=None
                     for term in objective.structure]
            feats = loDs_to_DoLs(feats)
            
            Z = dictApplyUFunc(lambda x: np.vstack(x).T, feats)
            if not all([np.all(np.isfinite(z)) for z in Z.values()]):
                objective._gram_super = None
                _loop_stats.record('EqRPS.gram_super_skip', 1, 1)
                return
            gram_mode = active_config().objectives.gram_mode
            if gram_mode == 'vcoef':
                objective._gram_super = VaryingCoefSetup.precompute_super(Z, sample_weights, grid_shapes,
                                                                          main_var=objective.main_var_to_explain)
            elif gram_mode == 'axis':  # backup
                objective._gram_super = GramSetup.precompute_super(Z, sample_weights, grid_shapes)
            else:
                # Basis-free instability metric: no Gram is needed, but the
                # cached Z still earns its keep -- it is what lets the sweep
                # slice out each candidate's target/features without
                # re-evaluating the terms.
                objective._gram_super = {'mode': None, 'Z': Z}
            _loop_stats.record('EqRPS.gram_super_built', 1, 1)
        except Exception as exc:
            # Defensive: any unexpected failure (shape mismatch, missing
            # cache) means we fall back -- numerics are preserved, only the
            # speedup is lost. Reported ONCE per process rather than
            # swallowed: a permanently-throwing fast path is invisible
            # otherwise (it cost every sweep of the multisample port, where
            # the dict of inner shapes was iterated as if it were a tuple).
            objective._gram_super = None
            _loop_stats.record('EqRPS.gram_super_skip', 1, 1)
            global _SUPER_GRAM_FAILURE_REPORTED
            if not _SUPER_GRAM_FAILURE_REPORTED:
                _SUPER_GRAM_FAILURE_REPORTED = True
                warnings.warn(
                    'EqRightPartSelector._precompute_super_gram: the tier-3 '
                    f'super-Gram fast path is disabled ({type(exc).__name__}: '
                    f'{exc}). Falling back to the per-target path for the rest '
                    'of this run.')

    def set_suboperators(self, operators: dict, probas: dict = {}):
        """Wire the suboperators, then tell ``coeff_calc`` what scale the
        support decision it will be handed was fitted on.

        The sparsity operator declares this about itself
        (``fits_physical_scale``); pushing it across here once per run keeps the
        two strategy builders -- which already hand this operator both
        suboperators together -- unaware of the coupling, and keeps
        ``coeff_calc`` from having to reach back into the equation for a marker.
        """
        super().set_suboperators(operators, probas)
        sparsity = operators.get('sparsity')
        coeff_calc = operators.get('coeff_calc')
        if sparsity is not None and coeff_calc is not None:
            coeff_calc.sparsity_fits_physical = bool(
                getattr(sparsity, 'fits_physical_scale', False))

    def _fit_and_score(self, objective: Equation, subop_args: dict):
        """Fit ``objective`` for its CURRENT ``target_idx``, then score it.

        This operator owns the fit. ``sparsity`` decides the support and
        ``coeff_calc`` puts physical magnitudes on it; only then is the
        fitness host -- a pure scorer that neither fits nor prunes -- asked
        for the candidate's discrepancy. ``None`` comes back for a candidate
        the host declines (all-zero support, a discrepancy over the degeneracy
        threshold, or an amplified near-null fit).

        The two ``force_out_of_place`` call sites (the term-sweep and the
        exit-guarantee probe) must stay identical, which is why they share
        this helper rather than repeating the three-step sequence.
        """
        self.suboperators['sparsity'].apply(objective, subop_args['sparsity'])
        self.suboperators['coeff_calc'].apply(objective, subop_args['coeff_calc'])
        return self.suboperators['fitness_calculation'].apply(
            objective, arguments=subop_args['fitness_calculation'],
            force_out_of_place=True)

    @_loop_stats.timed('EqRPS.apply')
    @HistoryExtender('\n -> The equation structure was detected: ', 'a')
    def apply(self, objective : Equation, arguments : dict,
              banned_sigs=frozenset()):
        """Select a right-part term for ``objective`` in-place.

        ``banned_sigs``: standalone-term signatures the terms this operator
        regenerates may not take -- inside a system, what the other equations'
        targets reserve (passed by ``SoEqRightPartSelector``; empty for a
        single equation). It reaches the regenerations inside
        ``simplify_equation`` and, via ``_redraw_banned_terms``, the
        ``Equation`` generators driven below (``randomize``,
        ``restore_property``), which are otherwise blind to the system.

        Handles two recoverable failure modes inside the outer loop via
        ``objective.randomize()`` (cheap, single-equation reroll) rather
        than bubbling up to the chromosome-level offspring loop -- the
        chromosome regen path is ~100x more expensive than a single
        equation reroll and floods the EA when many candidates fail.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # Duplicate-term detection: a frozenset of per-term factor signatures
        # has the same length as ``structure`` iff every term is distinct.
        # Comparing against ``terms_labels`` here would be dimensionally wrong
        # (see the same family of bugs fixed in ``enforce_rps_uniqueness`` and
        # ``simplify_equation``).
        signatures = {term.factors_labels for term in objective.structure}
        assert len(signatures) == len(objective.structure), \
            'Equation has duplicate terms; randomize before right-part selection.'

        outer_max_iter = 50
        inner_max_iter = 100
        outer_attempts = 0
        # Loop control, deliberately LOCAL. These two were Equation attributes;
        # persisting them meant an offspring could inherit "already simplified /
        # right part already correct", skip this loop entirely -- including the
        # ``reset_state(True)`` on its first line -- and leave RPS still carrying
        # its parent's weights. As locals they start False on every entry, so
        # the body always runs at least once and the equation is always refitted
        # for a target this invocation chose.
        simplified = False
        correct_right_part = False
        while not (simplified and correct_right_part):
            outer_attempts += 1
            if outer_attempts > outer_max_iter:
                warnings.warn(
                    'EqRightPartSelector.apply: outer loop did not converge '
                    f'after {outer_max_iter} iterations; accepting current state.'
                )
                break
            objective.reset_state(True)
            min_fitness = np.inf
            # One slot per non-target term plus the trailing intercept.
            weights_internal = np.zeros(len(objective.structure))
            min_idx = 0
            inner_attempts = 0
            # ``restore_property(deriv=True)`` injects a derivative-family
            # token into the structure; it's a refinement op, not a regen
            # signal. The randomize() fallback below would only fire if the
            # 200-iter restore_property loop failed 100 times in a row -- a
            # ~20 000-attempt impossibility in practice.
            while not any(term.contains_deriv(objective.main_var_to_explain) for term in objective.structure):
                inner_attempts += 1
                if inner_attempts > inner_max_iter:
                    warnings.warn(
                        'EqRightPartSelector.apply: restore_property failed to '
                        f'introduce a deriv of {objective.main_var_to_explain!r} '
                        f'after {inner_max_iter} attempts; randomizing equation.'
                    )
                    objective.randomize()
                    _redraw_banned_terms(objective, list(objective.structure),
                                         banned_sigs)
                    break
                _before = list(objective.structure)
                objective.restore_property(mandatory_family=False, deriv=True)
                # The injected term carries a derivative of THIS equation's
                # variable, so it can only collide with another equation's
                # target when that target is a composite carrying one of each
                # (the shared-target case) -- rare, but the same ban applies.
                # Scrubbing it here, before the ``while`` re-tests, keeps the
                # loop's own invariant: if the redraw took the derivative away,
                # the next attempt restores it.
                _redraw_banned_terms(
                    objective,
                    [t for t in objective.structure
                     if not any(t is b for b in _before)],
                    banned_sigs)
            _loop_stats.record('EqRPS.inner_derivative', inner_attempts, inner_max_iter)

            # Tier 3: precompute the super-Gram over all terms ONCE per
            # outer iter so the term-sweep below derives per-target
            # GramSetup views via pure slicing instead of rebuilding the
            # windowed XTWX matmul for every candidate.
            self._precompute_super_gram(objective)

            with _loop_stats.timer('EqRPS.term_sweep'):
                for target_idx, target_term in enumerate(objective.structure):
                    if not objective.structure[target_idx].contains_deriv(objective.main_var_to_explain):
                        continue
                    objective.target_idx = target_idx
                    fitness = self._fit_and_score(objective, subop_args)
                    if fitness is not None and fitness < min_fitness:
                        min_fitness = fitness
                        min_idx = target_idx
                        weights_internal = objective.weights_internal
                        weights_final = objective.weights_final
                        sw_weights = objective._cached_sw_weights
                        vc_score = objective._cached_vc_score

                    objective.weights_internal_evald = False
                    objective.weights_final_evald = False

            if np.isinf(min_fitness):
                # Every eligible target produced inf fitness for the
                # post-restore structure -- reroll this single equation
                # locally (cheap) and continue the outer loop.
                _loop_stats.record('EqRPS.inf_fitness_regen', 1, 1)
                objective.randomize()
                # A whole-equation reroll draws blind to the system; redraw
                # whatever landed on another equation's reserved signature
                # before the outer loop refits it.
                _redraw_banned_terms(objective, list(objective.structure),
                                     banned_sigs)
                continue

            objective.weights_internal = weights_internal
            objective.weights_final = weights_final
            objective._cached_sw_weights = sw_weights
            objective._cached_vc_score = vc_score
            objective.weights_internal_evald = True
            objective.weights_final_evald = True
            objective.target_idx = min_idx

            if not self.simplify_equation(objective, banned_sigs):
                simplified = True
            if objective.target is not None and objective.target.contains_deriv(objective.main_var_to_explain):
                correct_right_part = True

        _loop_stats.record('EqRPS.outer', outer_attempts, outer_max_iter)
        # Drop the super-Gram so a downstream consumer (e.g. fitness
        # recomputation outside the term-sweep) falls back to the
        # per-target ``GramSetup.__init__`` path; the cached super-Gram
        # is only valid for the structure observed during the sweep.
        objective._gram_super = None
        objective.right_part_selected = True
        # Exit guarantee: a valid target MUST leave RPS. The cap-break path
        # (outer loop exhausted) or a structural reroll inside the loop
        # (``objective.randomize()`` on inf-fitness) can leave the identity-
        # tracked target as None; install a deterministic, valid target so the
        # downstream ``Equation.evaluate`` never indexes a stale/None position.
        # Candidates: terms carrying a derivative of the explained variable
        # (the only physically valid right parts); fall back to term 0. Among
        # them, PREFER the first whose sweep-style fit is not declined by the
        # host (a None fit = zeroed / degenerate / amplified-identity, one
        # unified semantics) -- the exit guarantee must not hand a declined
        # candidate a free pass around the sweep. If every candidate is
        # declined, the first deriv-carrying term is installed anyway (a
        # target must leave RPS) and the in-place fitness backstop condemns
        # the amplified fit to LOSS_NAN_VAL (SolverFreeFitness.apply).
        if objective.target is None and objective.structure:
            deriv_idxs = [i for i, term in enumerate(objective.structure)
                          if term.contains_deriv(objective.main_var_to_explain)]
            # This is the one place the operator can install a leak without
            # drawing anything: the term it promotes may be what another
            # equation's target reserves. Order the candidates so unreserved
            # ones are tried first -- a stable sort, so structure order still
            # decides within each group, and the fit probe below still outranks
            # the ban (a reserved-but-fitting target beats an unreserved one
            # the host declines; the SoEq loop repairs the former, nothing
            # repairs a degenerate fit). If EVERY candidate is reserved, one is
            # installed anyway: a target must leave RPS.
            if banned_sigs:
                deriv_idxs.sort(key=lambda i: objective.structure[i]
                                .factors_labels in banned_sigs)
            chosen = None
            # Resolved once per apply, not once per candidate in the sweep.
            amplification_cap = active_config().search_space.rps_amplification_cap
            if amplification_cap is not None:
                for cand_idx in deriv_idxs:
                    objective.target_idx = cand_idx
                    cand_fit = self._fit_and_score(objective, subop_args)
                    objective.weights_internal_evald = False
                    objective.weights_final_evald = False
                    if cand_fit is not None:
                        chosen = cand_idx
                        break
            if chosen is None:
                chosen = deriv_idxs[0] if deriv_idxs else 0
            objective.target_idx = chosen
            _loop_stats.record('EqRPS.exit_target_fallback', 1, 1)
        # EXIT CONTRACT: an equation leaves this operator FITTED, for the
        # target that was actually installed. The probe above clears the
        # weight flags after each candidate it tries and the ``outer_max_iter``
        # break can leave them down too, so without this the scorer would be
        # handed an equation with no support decision -- which it now refuses
        # (SolverFreeFitness.apply) instead of quietly re-sparsifying, as the
        # retired ``needs_sparsity`` fallback used to. Refitting here also
        # repairs a real desync on that path: ``chosen`` could be installed
        # while the weights still described a DIFFERENT candidate target.
        #
        # BOTH flags are tested. The scorer reads ``weights_final``, and a
        # structural change mid-loop (``simplify_equation``, the linear-
        # dependence scrub) drops the fit wholesale; on the cap-break path that
        # left the support decision up with no coefficients behind it, which the
        # old internal-only test read as "already fitted".
        if not (getattr(objective, 'weights_internal_evald', False)
                and getattr(objective, 'weights_final_evald', False)):
            self._fit_and_score(objective, subop_args)
        objective.remove_zero_terms()
        objective.assert_state_invariants('EqRightPartSelector.apply exit')
        # Hard invariant: no duplicate terms may leave RPS. simplify and
        # scrub both regenerate-then-drop, so a surviving duplicate is a
        # logic error to surface HERE -- not one generation later at the
        # crossover/mutation assert (the crash site that "lies").
        _final_sigs = {term.factors_labels for term in objective.structure}
        assert len(_final_sigs) == len(objective.structure), \
            'EqRightPartSelector.apply: duplicate terms survived RPS.'

    def simplify_equation(self, objective: Equation, banned_sigs=frozenset()):
        # Get nonzero terms
        tgt = objective.target_idx
        # ``[:-1]`` drops the trailing intercept slot explicitly. Without it the
        # mask is one longer than ``nonrs_terms`` and only the ``zip`` below
        # hides the mismatch.
        nonzero_terms_mask = np.array([False if weight == 0 else True
                                       for weight in objective.weights_internal[:-1]], dtype=np.int32)
        nonrs_terms = [term for i, term in enumerate(objective.structure) if i != tgt]
        nonzero_terms = [item for item, keep in zip(nonrs_terms, nonzero_terms_mask) if keep]
        nonzero_terms.append(objective.target)
        equation_terms = [term.factors_labels_without_power for term in nonzero_terms]

        if len(equation_terms) <= 1:
            return False

        # Ratio-fit scrub: a nonzero feature term related to the target by
        # DIVISION in either direction (see ``_term_divides``) makes the
        # ``Lambda * g * (true identity)`` FD-error soak representable:
        #
        # * a COMPONENT (term divides target) -- the wave composite trap
        #   ``u*u_tt = 3.75*u_tt - 0.15*u_xx + 0.04*u*u_xx`` (g = 1), whose
        #   padded pair cancels exactly under the true relation;
        # * a MULTIPLE (target divides term) -- the hitchhiker shape
        #   ``0.687*u_t*u_tt - 0.0275*u_t*u_xx`` riding on the canonical
        #   target ``u_tt`` (g = u_t; the coefficient ratio is again
        #   exactly the true 0.04).
        #
        # Scrubbing the target-side member breaks every such family (the
        # partner alone cannot cancel and dies at the refit -- measured
        # under both the chi2 and vcoef keep-rules). Component padding
        # also BLOCKS the common-factor cancellation below; scrubbing it
        # first lets the surviving reduced identity (which DOES share the
        # factor) collapse to its canonical form on a later pass.
        divisors = [term for term in nonzero_terms[:-1]
                    if _related_to_target(term, objective.target)]
        if divisors:
            def _not_soak_related(term_):
                return not _related_to_target(term_, objective.target)
            for term in divisors:
                status = _regen_or_drop_term(
                    objective, term, max_iter=100,
                    stats_name='simplify_equation.divisor_scrub',
                    extra_ok=_not_soak_related, banned_sigs=banned_sigs)
                if status in ('target', 'floor'):
                    # Could not scrub without degenerating the equation, but
                    # ``_regen_or_drop_term`` may already have randomized the
                    # term, so this is still a structural change: report it
                    # and let the outer RPS loop reset and re-select (the same
                    # rule as the common-factor branch below).
                    break
            objective.reset_for_structure_change()
            return True

        # Degree reduction: when a SINGLE non-target term remains, the
        # equation is ``coef * f = g`` with f, g products of powered
        # factors. If every factor power in BOTH f and g shares a common
        # divisor p >= 2, the whole equation is a p-th power -- take the
        # p-th root (divide every power by p; the coefficient is recomputed
        # downstream). E.g. ``c*(u_xx)^2 = (u_tt)^2`` -> ``sqrt(c)*u_xx = u_tt``.
        # Keeps the lowest-degree equivalent form so it is not penalised /
        # mistaken for a distinct higher-order structure.
        if len(nonzero_terms) == 2:
            powers, integral = [], True
            for term in nonzero_terms:
                for factor in term.structure:
                    for i in factor.params_description:
                        if factor.params_description[i]["name"] == "power":
                            p = factor.params[i]
                            if float(p) != int(p) or int(p) < 1:
                                integral = False
                            powers.append(int(p))
            if integral and powers:
                root = int(np.gcd.reduce(np.array(powers, dtype=int)))
                if root >= 2:
                    # p-th root: divide every factor power by the gcd,
                    # collapsing the equation to its lowest equivalent
                    # degree (c*(u_xx)^2=(u_tt)^2 -> sqrt(c)*u_xx=u_tt).
                    for term in nonzero_terms:
                        for factor in term.structure:
                            for i in factor.params_description:
                                if factor.params_description[i]["name"] == "power":
                                    factor.set_param(int(factor.params[i]) // root, idx=i)
                        term.resetSavedState()
                    # The reduction can collapse a survivor onto a zero-weight
                    # candidate already in the structure (u^2 -> u when a u
                    # term exists). Such a colliding copy is ALWAYS a
                    # zero-weight non-survivor -- two genuinely nonzero terms
                    # cannot collide via a p-th root unless they were already
                    # equal, which the entry assert forbids -- so drop the
                    # redundant copies outright (drop-immediately). The reduced
                    # low-degree form survives on the kept terms; no revert.
                    keep_ids = {id(t) for t in nonzero_terms}
                    kept_labels = {t.factors_labels for t in nonzero_terms}
                    redundant = [t for t in objective.structure
                                 if id(t) not in keep_ids
                                 and t.factors_labels in kept_labels]
                    for t in redundant:
                        _regen_or_drop_term(
                            objective, t, max_iter=0,
                            stats_name='simplify_equation.degree_reduction')
                    objective.reset_for_structure_change()
                    return True
        common_factors = list(frozenset.intersection(*equation_terms))
        if not common_factors:
            # No symbolic reduction applies -- run the numeric completion of
            # the duplicate-term invariant: exactly linearly dependent
            # nonzero feature columns (the generalization of a duplicate
            # term the label test cannot see, e.g. two symbolically distinct
            # products evaluating to proportional fields) make the active-
            # support OLS refit rank-deficient ("infinite solutions"), so
            # such terms are regenerated like duplicates. The target is
            # never touched: a target in the SPAN of independent features is
            # a perfect fit (a valid identity), not a degeneracy.
            return self._regenerate_dependent_terms(objective,
                                                    nonzero_terms[:-1],
                                                    banned_sigs)

        for common_factor in common_factors:
            # Min power across the matching factor in every nonzero term.
            min_order = np.inf
            for term in nonzero_terms:
                for factor in term.structure:
                    if factor.structural_label_without_power == common_factor:
                        if factor.cache_label[1][0] < min_order:
                            min_order = factor.cache_label[1][0]

            # Reduce order of common factor in every term; drop zero-power factors.
            for term in nonzero_terms:
                factors_simplified = []
                for factor in term.structure:
                    if factor.structural_label_without_power == common_factor:
                        for i, value in enumerate(factor.params_description):
                            if factor.params_description[i]["name"] == "power":
                                factor.set_param(factor.params[i] - min_order, idx=i)
                                if factor.params[i] == 0:
                                    factors_simplified.append(factor)
                            else:
                                continue
                term.structure = [factor for factor in term.structure if factor not in factors_simplified]
                term.resetSavedState()

            # Reduction collisions are ALWAYS against zero-weight
            # non-survivors: two NONZERO terms cannot collide by shedding a
            # factor they both carried (they would have been duplicates
            # before), so a colliding copy is dead structure -- drop the
            # copy, keep the reduced carrier (the degree-reduction policy
            # above). Regenerating the CARRIER instead -- the old policy --
            # scattered exact identities into random terms and broke the
            # canonical collapse (the wave ratio-fit chain).
            keep_ids = {id(t) for t in nonzero_terms}
            kept_labels = {t.factors_labels for t in nonzero_terms}
            redundant = [t for t in objective.structure
                         if id(t) not in keep_ids
                         and t.factors_labels in kept_labels]
            for t in redundant:
                _regen_or_drop_term(
                    objective, t, max_iter=0,
                    stats_name='simplify_equation.cancel_collision')

            # A term that consisted of nothing but the common factor is now
            # empty / non-meaningful: regenerate it; if the pool can't
            # yield a unique, meaningful replacement within the cap, DROP
            # it. A duplicate must never ride out of RPS -- see the exit
            # assert in ``apply``.
            for term in nonzero_terms:
                status = _regen_or_drop_term(
                    objective, term, max_iter=100,
                    stats_name='simplify_equation.replace_term',
                    banned_sigs=banned_sigs)
                if status in ('target', 'floor'):
                    # The offending term is the RPS target, or dropping it
                    # would degenerate the equation. There is nothing to
                    # decline: the factor powers were ALREADY reduced above,
                    # the zero-power factors ALREADY removed, and
                    # ``_regen_or_drop_term`` may have randomized the term up
                    # to ``max_iter`` times. None of that is revertible, so
                    # report the change and let the outer RPS loop reset and
                    # re-select -- which is what this comment always claimed
                    # and ``return False`` never did. Returning False here made
                    # the caller read "nothing to simplify", exit the loop, skip
                    # the exit-contract refit, and prune in ``remove_zero_terms``
                    # using coefficients that described the PRE-simplification
                    # structure.
                    objective.reset_for_structure_change()
                    return True

            # Structure changed: invalidate stale fitness /
            # weights / AIC caches while leaving RPS to the
            # caller's outer loop.
            objective.reset_for_structure_change()
            return True
        # Unreachable: the loop body above returns on every path.
        raise AssertionError('simplify_equation: common-factor loop fell through')

    def _regenerate_dependent_terms(self, objective: Equation,
                                    feature_terms, banned_sigs=frozenset()) -> bool:
        """Regenerate nonzero non-target terms whose evaluated columns are
        EXACTLY linearly dependent (incl. against the intercept).

        Such columns leave the active-support coefficient solve with
        infinitely many solutions (the observed lv / lorenz failure), so
        they are held to the same regenerate-n-then-drop policy as
        duplicate labels. Strictly machine-exact (``_LIN_DEP_RTOL`` on unit
        columns): near-collinear physics -- including valid analytical
        identities -- is never touched, in keeping with the
        no-collinearity-machinery principle.

        MULTISAMPLE SEMANTICS. The coefficient fit is run PER TRAJECTORY and
        the resulting coefficient vectors are averaged (``VWSRSparsity.apply``),
        so a rank-deficient solve in ONE sample already corrupts that sample's
        coefficients and thereby the average. A term is therefore flagged when
        its column is exactly dependent in ANY sample (union), and a candidate
        replacement is accepted only when it is independent in EVERY sample.
        Each sample keeps its OWN basis, normalised by its OWN column norm --
        the same convention ``_flag_dependent_columns`` uses -- and nothing is
        ever concatenated across samples: trajectories differ in length and
        grid, so a stacked residual would not be comparable to ``_LIN_DEP_RTOL``.

        Returns True iff the structure changed; the caller's outer RPS loop
        then re-runs the term sweep and re-checks (bounded by its own
        iteration cap).
        """
        if not feature_terms:
            return False

        cols = loDs_to_DoLs([term.evaluate() for term in feature_terms]) # grids=None
        cols: Dict[int, List[np.ndarray]] = dictApplyUFunc(lambda x: [np.asarray(elem, dtype=float).reshape(-1) for elem in x],
                                                           cols)

        scan: Dict[int, Tuple[List[int], List[np.ndarray]]] = dictApplyUFunc(_flag_dependent_columns, cols)
        if any([sc is None for sc in scan.values()]):
            # Non-finite column: dependence undecidable, mirror the
            # super-Gram skip rather than regenerating on a data problem.
            _loop_stats.record('simplify_equation.lin_dep_skip', 1, 1)
            return False

        # Per-sample flagged column indices and per-sample kept bases. The
        # indices address ``feature_terms``, which is sample-independent, so
        # they are directly unionisable; the bases are NOT (different lengths).
        bases: Dict[int, List[np.ndarray]] = {key: elem[1] for key, elem in scan.items()}
        flagged_union = sorted({idx for elem in scan.values() for idx in elem[0]})

        if not flagged_union:
            return False

        def _independent(term) -> bool:
            """True iff ``term``'s column is independent of the kept basis in
            EVERY sample (per-sample unit normalisation, no concatenation)."""
            col = dictApplyUFunc(lambda x: np.asarray(x, dtype=float).reshape(-1),
                                 term.evaluate()) # grids=None
            if set(col.keys()) != set(bases.keys()):
                return False
            for key, basis in bases.items():
                u = col[key]
                if not np.all(np.isfinite(u)):
                    return False
                nrm = np.linalg.norm(u)
                if nrm == 0.0:
                    return False
                u = u / nrm
                K = np.stack(basis, axis=1)
                coef, *_ = np.linalg.lstsq(K, u, rcond=None)
                if np.linalg.norm(u - K @ coef) < _LIN_DEP_RTOL:
                    return False
            return True

        changed = False
        for i in flagged_union:
            term = feature_terms[i]

            status = _regen_or_drop_term(
                objective, term, max_iter=100,
                stats_name='simplify_equation.lin_dep',
                extra_ok=_independent, banned_sigs=banned_sigs)
            if status == 'floor':
                # ``_regen_or_drop_term`` randomized this term up to max_iter
                # times looking for an independent replacement and left the last
                # candidate in place -- the structure DID change, even though no
                # replacement was accepted. Reporting False here made the caller
                # treat the equation as clean, exit the outer loop and prune on
                # coefficients describing the pre-scrub structure.
                changed = True
                break
            if status in ('regenerated', 'dropped', 'target'):
                # 'target' cannot occur today (``feature_terms`` excludes the
                # target), but it is a structural change like the others and
                # must not be silently swallowed by an exhaustive-looking test.
                changed = True
                if status == 'regenerated':
                    # The accepted replacement joins each sample's basis so the
                    # remaining flagged terms are checked against it too.
                    col = dictApplyUFunc(lambda x: np.asarray(x, dtype=float).reshape(-1),
                                         term.evaluate()) # grids=None
                    for key in bases.keys():
                        nrm = np.linalg.norm(col[key])
                        if nrm > 0.0:
                            bases[key].append(col[key] / nrm)
        if changed:
            objective.reset_for_structure_change()
        return changed

    def use_default_tags(self):
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}

        
class RandomRHPSelector(CompoundOperator):
    '''
    
    Operator for selection of the right part of the equation to emulate approximation of non-trivial function. 
    Works in the following manner: in a loop each term is considered as the right part, for this division the 
    fitness function value is calculated. The term, corresponding to the separation with the highest FF value is 
    saved as the correct right part. 
    
    THIS OPERATOR OWNS THE FIT. Each candidate target is sparsified, given
    coefficients and only then scored (``_fit_and_score``) -- the fitness
    hosts are pure scorers that neither fit nor prune. It is also the single
    place that prunes: the one ``remove_zero_terms`` at the end of ``apply``,
    after the sweep has chosen a winner. (That prune cannot move into the
    sparsity operator, which runs per candidate and must keep every
    candidate term.)

    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the CompoundOperator class
        key - str, value - instance of a class, inhereted from the CompoundOperator.
        Suboperators, performing tasks of equation processing. Three are
        present, applied in this order per candidate target: ``sparsity``
        (support selection), ``coeff_calc`` (physical magnitudes) and
        ``fitness_calculation`` (the discrepancy candidates are ranked by).

    Methods:
    -----------
    apply(equation)
        return None
        Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``

    
    '''
    key = 'RandomRightPartSelector'

    @HistoryExtender('\n -> The equation structure was detected: ', 'a')
    def apply(self, objective : Equation, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        if not objective.right_part_selected:
            term_selection = [term_idx for term_idx, term in enumerate(objective.structure)
                              if term.contains_deriv(variable = objective.main_var_to_explain)]

            if len(term_selection) == 0:
                idx = np.random.choice([term_idx for term_idx, _ in enumerate(objective.structure)])
                prev_term = objective.structure[idx]
                # Bounded retry + dedup check: never spin against a finite
                # token pool, never introduce a duplicate term (see
                # feedback-structure-dedup memory).
                max_iter = 100
                candidate_term = None
                attempts = 0
                for _ in range(max_iter):
                    attempts += 1
                    candidate_term = Term(pool = prev_term.pool, mandatory_family = objective.main_var_to_explain,
                                          max_factors_in_term = len(prev_term.structure),
                                          create_derivs = True)
                    if not candidate_term.contains_deriv(variable = objective.main_var_to_explain):
                        continue
                    sig = candidate_term.factors_labels
                    if any(j != idx and t.factors_labels == sig
                           for j, t in enumerate(objective.structure)):
                        continue
                    break
                else:
                    warnings.warn(
                        f'RandomRHPSelector: could not produce a unique deriv term '
                        f'for {objective.main_var_to_explain!r} after {max_iter} '
                        f'attempts; keeping last candidate (may duplicate).'
                    )
                _loop_stats.record('RandomRHPSelector.candidate_gen', attempts, max_iter)

                objective.structure[idx] = candidate_term
            else:
                idx = np.random.choice(term_selection)

            objective.target_idx = idx
            objective.reset_explaining_term(idx)
            objective.right_part_selected = True


    def use_default_tags(self):
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}


# Relative projection residual (on unit-normalized columns) below which an
# evaluated term column counts as EXACTLY linearly dependent on the columns
# before it. This is a machine-exactness test for rank deficiency ("infinite
# solutions" in the OLS/Gram solves), NOT a collinearity threshold: exact
# dependence projects to ~eps*conditioning, while even severely near-collinear
# physics (incl. valid analytical identities) sits many orders of magnitude
# above 1e-10.
_LIN_DEP_RTOL = 1e-10


def _flag_dependent_columns(cols, rtol: float = _LIN_DEP_RTOL) -> Tuple[List[int], List[np.ndarray]]:
    """Greedy exact-dependence scan over evaluated feature columns.

    Seeds the kept basis with the intercept column (the coefficient fits
    include one, so an exactly-constant term column is redundant with it),
    then walks ``cols`` in order: a column whose unit-normalized projection
    residual on the kept basis falls below ``rtol`` -- or an all-zero
    column -- is flagged as dependent; otherwise it joins the basis.
    Dependence is weight-invariant (a diagonal sample weighting preserves
    exact linear dependence), so the unweighted columns are checked.

    Returns ``(flagged_indices, basis)`` where ``basis`` is the list of
    kept unit columns (intercept first), or ``None`` when any column is
    non-finite -- the check is not applicable then (mirrors the
    ``_precompute_super_gram`` skip).
    """
    n = cols[0].size
    basis = [np.full(n, 1.0 / np.sqrt(n))]
    flagged = []
    for i, col in enumerate(cols):
        if not np.all(np.isfinite(col)):
            return None
        nrm = np.linalg.norm(col)
        if nrm == 0.0:
            flagged.append(i)
            continue
        u = col / nrm
        K = np.stack(basis, axis=1)
        coef, *_ = np.linalg.lstsq(K, u, rcond=None)
        if np.linalg.norm(u - K @ coef) < rtol:
            flagged.append(i)
        else:
            basis.append(u)
    return flagged, basis


def amplification_ratio(objective: Equation) -> float:
    """Amplification ratio of the CURRENT candidate right-part fit:

        A = sum_j |c_j| * ||col_j||  /  ||target col||

    over the nonzero non-target terms plus the fitted intercept
    (``weights_internal = [*term_coefs, intercept]``, the unified layout of
    ``Equation._validate_weight_layout``). The intercept's design column is the
    constant ones-vector, so its contribution ``|c_0| * sqrt(n)`` is the SAME
    ``|c_j| * ||col_j||`` rule -- ``sqrt(n) == ||1||_2`` -- just written out
    because that column is never materialized. It used to be unreachable: the
    guard was ``len(w) > len(nonrs)``, false while the sparsity operators
    emitted a length-m ``weights_internal``, so every recorded A (truth anchors
    1.0-6.65 against the cap of 100.0) was measured WITHOUT it. A ~ 1 when the terms combine
    without cancellation to the target's magnitude (every truth anchor
    measures A in [1.0, 6.65]); A >> 1 is the amplified-identity parasite,
    where near-cancelling giant coefficients fit the target out of the
    noise of a valid near-null feature combination. Returns ``inf`` on a
    non-finite/degenerate evaluation so the caller declines the candidate.
    """
    # READ-ONLY evaluations: ``grids=()`` (non-None) makes Term.evaluate skip
    # its tensor_cache.add while serving cache hits normally and computing
    # misses identically (the Term-level override never forwards ``grids`` to
    # the actual computation). The ratio is a pure diagnostic -- it must not
    # mutate cache state (adds/evictions/saved flags), or its extra
    # evaluations perturb later fits at float precision and break run
    # reproducibility (observed as 1e-7-level legacy A/B drift when the ratio
    # started running for every sweep candidate).
    w = np.asarray(objective.weights_internal, dtype=float)
    if not np.all(np.isfinite(w)):
        return np.inf
    tgt = objective.target_idx
    t = dictApplyUFunc(lambda x: np.asarray(x, dtype=float).reshape(-1), 
                       objective.structure[tgt].evaluate()) # grids=()
    den = dictApplyUFunc(np.linalg.norm, t)
    if (not all(list(dictApplyUFunc(np.isfinite, den).values())) or
        any([d == 0.0 for d in den.values()])):
        return np.inf
    nonrs = [term for i, term in enumerate(objective.structure) if i != tgt]

    num = defaultdict(float) # 0.0
    for wj, term in zip(w[:-1], nonrs):
        if wj == 0.0:
            continue
        col = dictApplyUFunc(lambda x: np.asarray(x, dtype=float).reshape(-1), term.evaluate()) # grids=()
        num = dictAdd(num, dictApplyUFunc(lambda x: abs(wj) * np.linalg.norm(x), col), suppressed=True)

    if w[-1] != 0.0:
        # Intercept column of ones: its norm is sqrt(n) PER SAMPLE, and ``t`` is
        # the dict of per-sample target columns -- not a single array.
        num = dictAdd(num, dictApplyUFunc(lambda x: abs(w[-1]) * np.sqrt(x.size), t),
                      suppressed=True)

    num = dictApplyUFunc(np.divide, num, den)

    return np.mean(list(num.values()))


def _term_divides(term, target_term) -> bool:
    """True iff ``term`` DIVIDES ``target_term``: every factor of ``term``
    appears in ``target_term`` (same ``structural_label_without_power``)
    with at least the same power (``cache_label[1][0]``, the
    ``simplify_equation`` power convention).

    Used by the ratio-fit scrub in ``simplify_equation``: a divisor
    feature spans an exact algebraic component of the target (``u_tt``
    inside the composite target ``u * u_tt``), which is what makes the
    ``Lambda * (true identity)`` FD-error soak representable. Measured on
    the wave pool: with the divisor excluded, the leftover padding dies
    under both the chi2 and vcoef keep-rules, leaving the reduced
    identity that common-factor cancellation collapses to the canonical
    form."""
    for f in term.structure:
        matched = False
        for g in target_term.structure:
            if (g.structural_label_without_power
                    == f.structural_label_without_power
                    and g.cache_label[1][0] >= f.cache_label[1][0]):
                matched = True
                break
        if not matched:
            return False
    return True


def _related_to_target(term, target_term) -> bool:
    """True iff ``term`` is a COMPONENT of ``target_term`` (divides it) or a
    MULTIPLE of it (is divided by it) -- exactly the terms the ratio-fit scrub
    in ``simplify_equation`` regenerates."""
    return _term_divides(term, target_term) or _term_divides(target_term, term)


def _related_to_a_possible_target(term, equation) -> bool:
    """True iff ``term`` is related (``_related_to_target``) to the equation's
    target or to any other term the next right-part selection could make the
    target: those carrying a derivative of the explained variable, the sweep's
    own eligibility test in ``EqRightPartSelector.apply``.

    A repair is always followed by a fresh selection, so testing the current
    target alone lets through a term the scrub removes under the target chosen
    next (measured on lv_o2: ``dv/dx0 * d2u/dx0^2`` passes against a
    ``du/dx0`` target, and the sweep then picks ``d2u/dx0^2``). The point is
    the repaired term itself: scrubbing it throws away the coupling the repair
    preserved, and the redraw is a random term rather than the leaked quantity
    demoted to a factor. What this cannot foresee is ``simplify_equation``'s
    own algebra (common-factor cancellation, degree reduction) reshaping an
    accepted term.
    """
    return any(_related_to_target(term, other)
               for other in _possible_targets(equation, term))


def _possible_targets(equation, term):
    """The equation's target and every other term the next right-part
    selection could make the target -- those carrying a derivative of the
    explained variable (the sweep's eligibility test) -- excluding ``term``."""
    main_var = getattr(equation, 'main_var_to_explain', None)
    target = equation.target
    return [other for other in equation.structure
            if other is not term
            and (other is target or other.contains_deriv(main_var))]


def _power_index(factor) -> int:
    return next(idx for idx, info in factor.params_description.items()
                if info['name'] == 'power')


def _symbolic_reductions(term, other, max_steps: int = 16):
    """The factor signatures of ``term`` and ``other`` after
    ``simplify_equation``'s symbolic reductions of the two-term equation
    ``term = other``, run to a fixed point on clones: degree reduction (every
    power shares a divisor >= 2) and common-factor cancellation (a factor base
    in both terms loses the smaller of its two powers). A side reduced to
    nothing is ``None`` -- it would be regenerated, not kept. The scrub and the
    linear-dependence check are not modelled."""
    sides = [[f.copy_for_power_update() for f in term.structure],
             [f.copy_for_power_update() for f in other.structure]]
    for _ in range(max_steps):
        powers = [f.params[_power_index(f)] for side in sides for f in side]
        if powers and all(float(p) == int(p) and int(p) >= 1 for p in powers):
            root = int(np.gcd.reduce(np.array([int(p) for p in powers], dtype=int)))
            if root >= 2:
                for side in sides:
                    for f in side:
                        idx = _power_index(f)
                        f.set_param(int(f.params[idx]) // root, idx=idx)
                continue
        common = ({f.structural_label_without_power for f in sides[0]}
                  & {f.structural_label_without_power for f in sides[1]})
        if not common:
            break
        base = min(common, key=repr)
        order = min(f.params[_power_index(f)] for side in sides for f in side
                    if f.structural_label_without_power == base)
        for k, side in enumerate(sides):
            kept = []
            for f in side:
                if f.structural_label_without_power == base:
                    idx = _power_index(f)
                    f.set_param(f.params[idx] - order, idx=idx)
                    if f.params[idx] == 0:
                        continue
                kept.append(f)
            sides[k] = kept
        if not sides[0] or not sides[1]:
            break
    return tuple(frozenset(f.structural_label for f in side) if side else None
                 for side in sides)


def _undone_by_simplify(term, equation, banned_sigs) -> bool:
    """True iff ``simplify_equation``'s own reductions of ``term`` against the
    equation's CURRENT target turn either side into one of ``banned_sigs``.

    The pair is the case the refit really does leave: ``term`` alone with the
    target. E.g. the leak ``dv/dx0`` wrapped as ``u*dv/dx0`` under the target
    ``u*du/dx0`` cancels ``u`` back off, and ``(dv/dx0)^2`` under
    ``(du/dx0)^2`` loses the square to degree reduction -- both restore the
    very leak the repair removed. The other side counts too: ``v*du/dx0``
    under ``du/dx0*dv/dx0`` cancels the TARGET down to the leak. Those
    reductions are correct (the wrapped law IS the leaked law times the
    shared factor), so it is the repair that has to avoid them.

    Only a side the reductions CHANGED counts. An equation may legitimately
    carry a banned signature already -- its own target, when two equations
    share one -- and reading that back unchanged would veto every draw.

    Deliberately the current target only, unlike the scrub rule
    (``_related_to_a_possible_target``): the scrub acts on whatever target is
    chosen next, but a cancellation under another candidate needs the refit
    to zero every term that lacks the factor. Checking every candidate as if
    it would (measured on the double pendulum) vetoed the true coupling
    ``cos(D)*th2''`` whenever a ``cos(D)*th1'`` term was present, although the
    real refit never cancelled it; the rarer switched-target cancellations
    measured on LV are left to the SoEq convergence loop.
    """
    target = equation.target
    if target is None or target is term:
        return False
    before = (term.factors_labels, target.factors_labels)
    return any(sig in banned_sigs and sig != was
               for sig, was in zip(_symbolic_reductions(term, target), before))


def _regen_or_drop_term(equation: Equation, term, *, max_iter: int = 100,
                        min_terms: int = 2,
                        stats_name: str = 'simplify_equation.regen_or_drop',
                        extra_ok=None, banned_sigs=frozenset()) -> str:
    """Make ``equation.structure`` unique w.r.t. ``term`` by regenerating
    ``term`` up to ``max_iter`` times; if it is still empty / non-meaningful
    / a duplicate, DROP it from the structure.

    ``extra_ok`` (optional ``Term -> bool``) extends the acceptability
    predicate beyond label uniqueness -- used by the linear-dependence
    scrub, whose redundant columns are label-unique yet must still be
    regenerated (and dropped on cap-hit, like a persistent duplicate).

    ``banned_sigs`` are signatures a fresh draw may not take, nor be reduced
    onto by ``simplify_equation`` (``_undone_by_simplify``): inside a system,
    the standalone terms other equations' targets reserve
    (``SoEqRightPartSelector``). It constrains draws only -- a term that
    already carries one is not regenerated for it here; that leak is the SoEq
    repair's, which first tries to keep the quantity as a factor.

    This is the simplify/scrub cap-hit policy -- *regenerate-n-then-drop* --
    deliberately distinct from the *keep-or-revert* ``retry_until_unique``
    policy used by ``Equation.__init__`` and the mutation operators.
    ``max_iter == 0`` means "drop immediately if unacceptable" (no
    regeneration) -- used by the degree-reduction branch and the scrub
    duplicate gate.

    The acceptability predicate ranges over the FULL structure, so a
    duplicate against a zero-weight candidate elsewhere is caught too. The
    RPS target is never dropped: if ``term`` is the target AND a duplicate,
    the OTHER member of its duplicate group is dropped instead; if the
    target is merely empty/non-meaningful, ``'target'`` is returned for the
    caller to handle. Refuses to drop below ``min_terms`` (returns
    ``'floor'``). On a drop, ``target_idx`` is reindexed exactly as in
    ``Equation.remove_zero_terms``.

    Returns one of ``'ok'`` (already acceptable), ``'regenerated'``,
    ``'dropped'``, ``'target'``, ``'floor'``.
    """
    idx = next((j for j, t in enumerate(equation.structure) if t is term), None)
    if idx is None:
        return 'ok'  # already dropped earlier in this pass

    def _acceptable(fresh: bool = False):
        if len(term.structure) == 0 or not term.contains_meaningful():
            return False
        signatures = {t.factors_labels for t in equation.structure}
        if len(signatures) != len(equation.structure):
            return False
        if fresh and banned_sigs and (
                term.factors_labels in banned_sigs
                or _undone_by_simplify(term, equation, banned_sigs)):
            return False
        return extra_ok is None or bool(extra_ok(term))

    cap = max_iter if max_iter > 0 else 1
    if _acceptable():
        _loop_stats.record(stats_name, 1, cap)
        return 'ok'

    attempts = 0
    for _ in range(max_iter):
        attempts += 1
        term.randomize()
        term.resetSavedState()
        if _acceptable(fresh=True):
            _loop_stats.record(stats_name, attempts, cap)
            equation._invalidate_label_cache()
            return 'regenerated'
    _loop_stats.record(stats_name, max(attempts, 1), cap)

    # Exhausted (or max_iter == 0): drop the offending term, if legal.
    target_term = equation.target          # the Term, or None
    drop_term = term
    if target_term is not None and term is target_term:
        # Can't drop the RPS target. If it duplicates another term, drop
        # that other (non-target) member; if it is merely empty/non-
        # meaningful, leave it for the caller to resolve.
        my_label = term.factors_labels
        other = next((t for t in equation.structure
                      if t is not term and t.factors_labels == my_label), None)
        if other is None:
            equation._invalidate_label_cache()
            return 'target'
        drop_term = other
    if len(equation.structure) <= min_terms:
        equation._invalidate_label_cache()
        return 'floor'
    # Capture the dropped term's weight slot BEFORE mutating the structure.
    drop_idx = next((j for j, t in enumerate(equation.structure) if t is drop_term), None)
    tgt_idx = equation.target_idx
    equation.structure = [t for t in equation.structure if t is not drop_term]
    # No manual target_idx decrement: the identity-tracked target auto-tracks
    # the surviving target Term (``drop_term`` is never the target -- the
    # branch above reroutes a target collision to the other duplicate).
    # Drop the matching weight slot from both vectors so they keep describing
    # the structure they are indexed against (Equation._validate_weight_layout).
    # Leaving them one slot too long is mostly masked -- the RPS outer loop
    # re-runs and ``reset_state(True)`` nulls them -- but not on the cap-break
    # path, where the stale vectors flow straight into ``remove_zero_terms``
    # off by one for every term past the dropped one.
    if drop_idx is not None and tgt_idx is not None and drop_idx != tgt_idx:
        wpos = equation.weight_index(drop_idx, tgt_idx)
        if getattr(equation, 'weights_internal_evald', False):
            equation.weights_internal = np.delete(np.asarray(equation.weights_internal), wpos)
        if getattr(equation, 'weights_final_evald', False):
            equation.weights_final = np.delete(np.asarray(equation.weights_final), wpos)
    equation._invalidate_label_cache()
    return 'dropped'


def _redraw_banned_terms(equation: Equation, terms, banned_sigs,
                         stats_name: str = 'EqRPS.generator_ban_scrub') -> bool:
    """Redraw each of ``terms`` whose signature is reserved by another
    equation of the system, the way the regenerations inside
    ``simplify_equation`` already avoid ``banned_sigs``.

    ``terms`` are the ones a generator on ``Equation`` JUST created. Those
    generators -- ``randomize`` (a whole-equation reroll) and
    ``restore_property`` (a derivative injection) -- belong to the equation,
    not to the system, so they draw from the pool with no idea what the other
    equations explain and land on a reserved signature freely. Measured on
    Lotka-Volterra: 94 of 478 rerolls inside ``EqRightPartSelector.apply``
    built another equation's target as a standalone term, and 61 of those
    rode out of the operator for the SoEq convergence loop to repair a pass
    later -- 55 of them drawn INSIDE that loop, i.e. the re-selection after a
    repair putting the leak straight back.

    Freshly drawn terms only, never inherited structure. A leak an equation
    arrived with may be real coupling, which the SoEq repair demotes to a
    FACTOR (``_wrap_term_with_factor``) rather than destroying; scrubbing it
    here would throw that away. A term drawn at random a moment ago carries
    no such information, so redrawing it costs nothing -- which is also why
    this does not wrap.

    Returns True if any term was redrawn or dropped.
    """
    if not banned_sigs:
        return False

    def _not_banned(term_):
        return term_.factors_labels not in banned_sigs

    changed = False
    for term in terms:
        if term.factors_labels not in banned_sigs:
            continue
        # ``_regen_or_drop_term`` reads the ban on fresh DRAWS only, so the
        # already-banned term in hand needs ``extra_ok`` to fail the entry
        # test and start the redraw; the cap-hit drop policy then applies.
        status = _regen_or_drop_term(equation, term, max_iter=100,
                                     stats_name=stats_name,
                                     extra_ok=_not_banned,
                                     banned_sigs=banned_sigs)
        if status != 'ok':
            changed = True
    return changed


def _target_term_in_other_equation(eq_with_target: Equation,
                                   eq_other: Equation):
    """Return ``eq_with_target``'s TARGET-term factor signature iff that
    whole term also appears as a (standalone) term in ``eq_other``'s ACTIVE
    structure; otherwise ``None``.

    This enforces target-term uniqueness across a system: the explained
    right-part term of one equation may not be carried as a complete term
    by another equation of the same system (the documented Lotka-Volterra
    leak where ``dv/dx0`` rode into the ``u`` equation as its own term).

    It is deliberately a WHOLE-TERM equality test -- ``target.factors_labels``
    is one element of ``eq_other.active_terms_labels`` -- NOT a
    sub-product/divides test: a target derivative appearing only as a
    FACTOR inside a composite coupling term (e.g. continuity's ``v_y``
    inside the v-momentum ``v*v_y`` of true Navier-Stokes) is legitimate
    physics and is left untouched.

    Two signatures can match, and the one RETURNED is the signature of the
    offending term in ``eq_other`` -- the term the caller has
    ``_break_equation_duplication`` repair. For a bare target that is the
    target's own signature; for a decorated target caught by the deriv-core
    rule below it is the CORE's, since that is the term ``eq_other`` actually
    carries. Either way the caller bans everything ``eq_with_target``
    reserves (``_reserved_signatures``), so a repair never turns the core into
    the whole target or back.

    Directional by construction (``eq_with_target``'s target into
    ``eq_other``); the caller scans both orderings of every pair. The
    target term is re-fetched as ``structure[target_idx]`` (never cached
    across rerolls) and guarded exactly as the existing degeneracy path
    (right_part_selection.py:608-612). ACTIVE scope means a zero-weight
    padding copy is ignored; any later reactivation is caught by the next
    per-generation RPS pass.
    """
    tgt = eq_with_target.target
    if tgt is None:
        return None
    target_sig = tgt.factors_labels
    if target_sig in eq_other.active_terms_labels:
        return target_sig

    # DERIV-CORE extension. A DECORATED target reserves its derivative core
    # exactly as a bare target would: if the v-equation explains
    # ``dv/dx0 * cos``, a standalone ``dv/dx0`` in the u-equation is the same
    # leak, because the decoration is the search's own choice and not a
    # different quantity. Without this the costume is a bypass -- the
    # Lotka-Volterra coupled-junk pair works precisely by dressing the
    # v-target in ``cos``, which unlocks standalone ``v_t`` for the
    # u-equation's dominating sum/differentiated identities.
    #
    # Still WHOLE-TERM only, so the Navier-Stokes carve-out is untouched: a
    # core appearing as a FACTOR of a composite coupling term (``v*v_y``) is
    # legitimate physics and is not matched here.
    core_sig = _target_core_signature(tgt)
    if core_sig is not None and core_sig in eq_other.active_terms_labels:
        return core_sig
    return None


def _target_core_signature(target_term):
    """The standalone-term signature of a DECORATED target's derivative core,
    or ``None`` for a bare target or one without a unique genuine derivative
    factor (``_deriv_core_factor``). A bare target's core is the target
    itself, so it adds nothing."""
    if len(target_term.structure) <= 1:
        return None
    core = _deriv_core_factor(target_term)
    return frozenset((core.structural_label,)) if core is not None else None


def _reserved_signatures(equation) -> set:
    """Every standalone-term signature ``equation``'s target reserves across a
    system: the target itself and, for a decorated target, its derivative core
    (``_target_term_in_other_equation`` flags either in another equation)."""
    tgt = equation.target
    if tgt is None:
        return set()
    core_sig = _target_core_signature(tgt)
    return {tgt.factors_labels} | ({core_sig} if core_sig is not None else set())


def _deriv_core_factor(term: Term):
    """The one genuine derivative factor of ``term``, or ``None``.

    "Genuine" is the predicate ``Term.contains_deriv`` already applies
    (main_structures.py): a derivative factor carrying a real ``deriv_code``
    whose evaluator is the plain one, so a trig- or grid-decorated token is
    never mistaken for the derivative it decorates. Ambiguity -- no such
    factor, or more than one -- returns ``None`` rather than a guess, which
    is the same "exactly one" rule ``contains_deriv`` enforces.
    """
    cores = [factor for factor in term.structure
             if factor.is_deriv and factor.deriv_code != [None, ]
             and factor.evaluator._evaluator == simple_function_evaluator]
    return cores[0] if len(cores) == 1 else None


def _wrap_term_with_factor(equation: Equation, term: Term, banned_sigs,
                           max_tries: int = 20) -> bool:
    """Demote ``term`` (a leaked standalone copy of another equation's
    target) into a FACTOR of a composite term: multiply it by one extra
    pool factor instead of destroying it.

    The whole-term uniqueness invariant allows another equation's target
    as a factor inside a composite coupling term, just not as a separate
    term. The leaked standalone copy is often the small-angle/stepping-stone
    form of legitimate coupling physics (e.g. ``th2''`` inside the ``th1``
    double-pendulum equation, whose true form is ``cos(D)*th2''``), so
    wrapping preserves that information where a reroll erases it.

    Accepts the wrap iff the new signature (a) differs from the original,
    (b) is outside ``banned_sigs``, (c) duplicates no other term of the
    equation and (d) is not a component or multiple of the equation's target
    or of any term that could become it (``_related_to_a_possible_target``).
    The pool draw includes the equation's own derivative family, so without
    (d) the wrap could build e.g. ``dv/dx0 * du/dx0`` for a ``du/dx0`` target
    -- which the ratio-fit scrub regenerates on the very next RPS pass,
    throwing away the coupling this wrap exists to keep -- and (e) does not
    let ``simplify_equation``'s own algebra reduce it, or the target, back
    onto the leak or another banned signature (``_undone_by_simplify``):
    ``u*dv/dx0`` under a ``u*du/dx0`` target is just the leak with ``u``
    multiplied through, and cancelling it is correct.
    Restores the original structure and returns False when the per-term
    factor cap is already reached or no acceptable wrap was drawn -- the
    caller then falls back to the randomize repair.
    """
    # The authoritative factor cap is the equation-level metaparameter
    # (evolution-built terms mirror it, but e.g. translated equations
    # leave the Term attribute at its constructor default of 1).
    try:
        cap = equation.metaparameters['max_factors_in_term']['value']
    except (AttributeError, KeyError, TypeError):
        cap = term.max_factors_in_term
    if isinstance(cap, dict):
        cap = max(cap['factors_num'])
    if len(term.structure) >= int(cap):
        return False
    original = term.structure
    original_sig = term.factors_labels
    other_sigs = {t.factors_labels for t in equation.structure if t is not term}
    attempts = 0
    for _ in range(max_tries):
        attempts += 1
        try:
            _, factor = term.pool.create(label=None, create_meaningful=False)
        except ValueError:
            break
        # filter_powers clones via copy_for_power_update, so ``original``
        # stays intact for the restore path below.
        term.structure = filter_powers(list(original) + [factor])
        term.resetSavedState()
        sig = term.factors_labels
        if (sig != original_sig and sig not in banned_sigs and sig not in other_sigs
                and not _related_to_a_possible_target(term, equation)
                and not _undone_by_simplify(term, equation,
                                            set(banned_sigs) | {original_sig})):
            _loop_stats.record('break_equation_duplication.wrap', attempts, max_tries)
            return True
    term.structure = original
    term.resetSavedState()
    _loop_stats.record('break_equation_duplication.wrap_fail', attempts, max_tries)
    return False


def _break_equation_duplication(equation: Equation, shared_sigs, *,
                                preferred_sigs=(), forbidden_sigs=None,
                                max_iter: int = 2000) -> bool:
    """Break a system-level degeneracy by repairing ONE term of ``equation``
    whose factor signature belongs to ``shared_sigs`` -- the signatures this
    equation may not carry as standalone terms: everything another
    equation's target reserves (the target and, when decorated, its
    derivative core). The repaired term may not land on any of them either.

    A non-target term is repaired when there is one, the term matching one of
    ``preferred_sigs`` (typically the other equation's target signature)
    first, so the rerolled equation moves away from "the other equation's
    explained quantity" before touching genuinely shared coupling terms.
    Repair prefers demoting that term to a factor of a composite term
    (:func:`_wrap_term_with_factor`); only when that fails is it randomized
    away. When the equation's OWN target is the sole offender -- two
    equations sharing one composite target -- the target is randomized
    directly and the caller's re-selection picks a new one.

    The randomize loop demands that the replacement (a) leaves
    ``shared_sigs``, (b) does not duplicate another term, (c) is not a
    component or multiple of any possible target of the equation (the wrap's
    rule (d)) and (d) is not reduced back onto a banned signature by
    ``simplify_equation`` (the wrap's rule (e)) -- (d) for a rerolled TARGET
    only while the redraw still carries a derivative of the explained
    variable, i.e. only while re-selection could pick it again; see the branch
    below for why there is otherwise no pair to model. On cap-hit a surviving
    duplicate is dropped via the ``_regen_or_drop_term`` drop policy (a
    still-shared-but-unique term is tolerated -- the caller's convergence
    loop re-checks -- and a related one is left to the scrub).

    ``forbidden_sigs`` defaults to ``shared_sigs`` and is what the REPLACEMENT
    is judged against. ``shared_sigs`` already carries the partner equation's
    reservations, so what a caller adds here is every THIRD equation's: in a
    system of three or more, a repair must not land on one of those either.

    Returns True if the structure changed; cached fitness/weight state is
    reset on the way out so the caller can re-run right-part selection.
    """
    forbidden = frozenset(shared_sigs if forbidden_sigs is None else forbidden_sigs)
    candidates = [term for idx, term in enumerate(equation.structure)
                  if idx != getattr(equation, 'target_idx', None)
                  and term.factors_labels in shared_sigs]
    target = equation.target
    rerolling_target = False
    if candidates:
        preferred = [t for t in candidates if t.factors_labels in preferred_sigs]
        term = preferred[0] if preferred else candidates[0]

        # Demote-to-factor repair first: keep the leaked target alive as a
        # factor of a composite coupling term (legal under the whole-term
        # rule) instead of rerolling it into unrelated structure. Falls back
        # to the randomize path when the wrap cannot produce a unique term.
        if _wrap_term_with_factor(equation, term, set(forbidden)):
            equation.reset_for_structure_change()
            return True
    elif target is not None and target.factors_labels in shared_sigs:
        # The banned term is this equation's OWN target: two equations of the
        # system explain one composite target (``du/dx0 * dv/dx0`` carries a
        # derivative of each variable, so both may pick it). A target cannot
        # be demoted to a factor, so it goes straight to the reroll below;
        # the caller's re-selection then chooses a new target, and
        # ``restore_property`` supplies a derivative term if none is left.
        term = target
        rerolling_target = True
    else:
        return False

    attempts = 0
    for _ in range(max_iter):
        attempts += 1
        term.randomize()
        term.resetSavedState()
        signatures = {t.factors_labels for t in equation.structure}
        duplicate = len(signatures) != len(equation.structure)
        related = _related_to_a_possible_target(term, equation)
        if rerolling_target and term.contains_deriv(equation.main_var_to_explain):
            # A rerolled target that re-selection can pick again is judged the
            # way the scrub and simplify will judge it then: no other term may
            # be its component or multiple (the scrub would delete it), and no
            # pair with it may cancel onto a banned signature
            # (``_undone_by_simplify`` skips the target itself).
            others = [t for t in equation.structure if t is not term]
            related = related or any(_related_to_target(t, term) for t in others)
            undone = any(sig in forbidden and sig != was
                         for t in others
                         for sig, was in zip(_symbolic_reductions(t, term),
                                             (t.factors_labels, term.factors_labels)))
        else:
            # On the target-reroll path this branch is VACUOUS, by
            # construction rather than by oversight: ``_undone_by_simplify``
            # pairs a term with the equation's CURRENT target, and here the
            # term IS that target, so it returns False at once. A redraw that
            # dropped the main-var derivative cannot be chosen as the next
            # target, so there is no pair for it to model. What still guards
            # such a redraw is ``_related_to_a_possible_target`` above (the
            # scrub's rule, run unconditionally), and for the derivative
            # ``restore_property`` then injects, ``_redraw_banned_terms`` in
            # ``EqRightPartSelector.apply``. Measured on the LV pool: the
            # derivative survives 82/300 (plain) and 22/300 (adversarial)
            # target rerolls, and of the rerolls that skip the branch above,
            # none would have been reduced onto a banned signature against any
            # eligible next target -- a two-factor pool cannot cancel INTO a
            # two-factor composite.
            undone = _undone_by_simplify(term, equation, forbidden)
        if (term.factors_labels not in forbidden and not duplicate and not related
                and not undone):
            break
    _loop_stats.record('break_equation_duplication', attempts, max_iter)
    # Cap-hit may leave the rerolled term as a DUPLICATE -- a duplicate must
    # never ride out of RPS, so drop it (regenerate attempts already spent).
    _regen_or_drop_term(equation, term, max_iter=0,
                        stats_name='break_equation_duplication.drop')
    signatures = {t.factors_labels for t in equation.structure}
    if len(signatures) != len(equation.structure):
        # The drop was refused by the two-term floor. Uniqueness is the hard
        # invariant -- the next ``EqRightPartSelector.apply`` asserts on it --
        # while the ban and the relation rules are preferences, so redraw on
        # uniqueness alone rather than hand a duplicate to the caller.
        _regen_or_drop_term(equation, term, max_iter=100,
                            stats_name='break_equation_duplication.dedup')

    equation.reset_for_structure_change()
    return True


class SoEqRightPartSelector(CompoundOperator):
    """Chromosome-level RPS that prevents system-level degeneracy.

    Invariant: no equation's TARGET term may appear as a whole (standalone)
    term in ANY other equation of the system -- the explained right-part
    term of one equation is reserved to that equation. This is a whole-term
    equality rule, NOT a sub-product one: a target derivative appearing only
    as a FACTOR inside a composite coupling term of another equation (e.g.
    continuity's ``v_y`` inside the v-momentum convective term ``v*v_y`` of
    Navier-Stokes) is legitimate physics and is left untouched.

    Cross-equation FACTOR sharing is otherwise allowed: an equation for
    ``v`` may carry ``du/dx0`` inside a composite term even when the
    equation for ``u`` explains ``du/dx0`` -- only a bare standalone
    ``du/dx0`` term in the ``v`` equation is forbidden.

    This subsumes the old "no two equations share an identical active
    structure" guard: two equations that collapse onto the same law (e.g.
    both Navier-Stokes velocity equations becoming continuity) necessarily
    have DIFFERENT targets, so each carries the other's target as a
    standalone term and is broken here. Two equations can also share the
    IDENTICAL target -- a composite carrying a derivative of each main
    variable (``du/dx0 * dv/dx0``), measured on Lotka-Volterra and Lorenz in
    under 1% of calls -- and then the equation the ordered scan finds carrying
    the other's target (normally the later one) has its target rerolled and
    re-selected.

    Every per-equation selection is told what the other equations' targets
    reserve (``_reserved_signatures``), so the terms it regenerates while
    simplifying never take those signatures: in the forward pass, the
    equations already selected; in the convergence loop, all of them.

    Mechanics: a plain per-equation forward pass first, then a bounded
    convergence loop that, each pass, rerolls any equation carrying another
    equation's target term as a whole term (via
    ``_break_equation_duplication`` + right-part re-selection), and checks
    on the way out that none is left. The one
    exception is a target RPS's exit fallback installed (it carries no
    derivative of its own variable): there the equation carrying nothing is
    the one repaired, since its target is what has no right to the quantity.
    The loop exits at the first pass with no changes (fixed point).
    """
    key = 'SoEqRightPartSelector'

    @_loop_stats.timed('SoEqRPS.apply')
    def apply(self, objective, arguments: dict):
        """Run per-equation RPS, then resolve system degeneracies in-place.

        Failures inside ``EqRightPartSelector.apply`` are handled locally
        via per-equation ``objective.randomize()`` -- this method has no
        regen signal to forward.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)
        eq_selector = self.suboperators['eq_right_part_selector']
        eq_args = subop_args.get('eq_right_part_selector', arguments)

        equations = list(objective)

        def reserved_by_others(equation, others):
            banned = set()
            for other in others:
                if other is not equation:
                    banned |= _reserved_signatures(other)
            return frozenset(banned)

        # Forward pass: plain per-equation right-part selection. No
        # cross-equation scrubbing -- shared terms are legitimate coupling --
        # but regenerated terms avoid what the equations selected so far
        # reserve.
        for idx, equation in enumerate(equations):
            eq_selector.apply(objective=equation, arguments=eq_args,
                              banned_sigs=reserved_by_others(equation, equations[:idx]))

        # Degeneracy resolution: enforce target-term uniqueness until a full
        # pass makes no change or the pass budget is exhausted.
        max_passes = 50
        passes_used = 0
        leaks_seen = False
        for _ in range(max_passes):
            passes_used += 1
            any_changes = False
            # Whether THIS pass's scan still found a leak, repaired or not --
            # the postcondition below reads the last pass's value.
            leaks_seen = False
            # Target-term uniqueness: no equation's TARGET term may appear
            # as a whole (standalone) term in ANOTHER equation of the
            # system. Directional -- scan every ordered pair (i -> j) and,
            # when eq i's target rides in eq j, reroll eq j's offending copy
            # via the same _break_equation_duplication drop-or-reroll
            # primitive, then re-select eq j's right part. A target
            # derivative that appears only as a FACTOR inside a composite
            # coupling term of eq j (e.g. continuity's ``v_y`` in ``v*v_y``)
            # is left untouched -- this is whole-term equality, not
            # sub-product. A decorated target reserves its bare derivative
            # core on the same whole-term terms. If the only match is eq j's
            # own target (both equations explain one composite target),
            # _break_equation_duplication rerolls that target and eq j's
            # right part is re-selected below.
            for i in range(len(equations)):
                for j in range(len(equations)):
                    if i == j:
                        continue
                    eq_i, eq_j = equations[i], equations[j]
                    leak_sig = _target_term_in_other_equation(eq_i, eq_j)
                    if leak_sig is None:
                        continue
                    leaks_seen = True
                    if (eq_j.target is not None
                            and eq_j.target.factors_labels == leak_sig
                            and eq_j.target.contains_deriv(eq_j.main_var_to_explain)
                            and not eq_i.target.contains_deriv(eq_i.main_var_to_explain)):
                        # Only eq j explains the shared quantity legitimately:
                        # eq i's target carries no derivative of its own
                        # variable, so RPS's exit fallback installed it. Reroll
                        # THAT one instead -- the (j, i) scan cannot, because a
                        # decorated fallback target leaks only its core, which
                        # eq i does not carry as a standalone term.
                        # eq i carries no non-target copy of its own target
                        # signature (the entry assert forbids duplicates), so
                        # this takes the target path and always reports a
                        # change. ``reserved_by_others`` excludes eq i itself,
                        # hence its own target is added by hand.
                        _break_equation_duplication(
                            eq_i, {eq_i.target.factors_labels},
                            preferred_sigs={eq_i.target.factors_labels},
                            forbidden_sigs=reserved_by_others(eq_i, equations)
                            | {eq_i.target.factors_labels})
                        eq_i.right_part_selected = False
                        eq_selector.apply(objective=eq_i, arguments=eq_args,
                                          banned_sigs=reserved_by_others(eq_i, equations))
                        _loop_stats.record('SoEqRPS.fallback_target_repair', 1, 1)
                        any_changes = True
                        continue
                    # Ban everything eq i reserves, whichever part of it
                    # leaked: its whole target and, when decorated, its
                    # derivative core. Otherwise the repair can turn one into
                    # the other -- wrap ``dv/dx0`` into ``u*dv/dx0``, or
                    # randomize ``u*dv/dx0`` into ``dv/dx0`` -- and the next
                    # pass flags it again. The leaked term is still the one
                    # repaired.
                    banned = {leak_sig} | _reserved_signatures(eq_i)
                    # ``reserved_by_others`` already covers eq i (it excludes
                    # only eq j), so it is a superset of ``banned``; what it
                    # adds is every THIRD equation's reservations, which the
                    # replacement must avoid too.
                    changed = _break_equation_duplication(
                        eq_j, banned, preferred_sigs={leak_sig},
                        forbidden_sigs=reserved_by_others(eq_j, equations))
                    if not changed:
                        continue
                    # ``simplified`` / ``is_correct_right_part`` are locals of
                    # ``EqRightPartSelector.apply`` now, so re-invoking it is
                    # the whole mechanism -- it re-enters its loop from scratch.
                    eq_j.right_part_selected = False
                    eq_selector.apply(objective=eq_j, arguments=eq_args,
                                      banned_sigs=reserved_by_others(eq_j, equations))
                    _loop_stats.record('SoEqRPS.target_leak_repair', 1, 1)
                    any_changes = True

            if not any_changes:
                break
        _loop_stats.record('SoEqRPS.degeneracy_passes', passes_used, max_passes)

        # POSTCONDITION. The loop exits either at a fixed point or with its
        # pass budget spent, and neither state was checked. Both can in
        # principle leave a leak standing: the budget can run out, and a pass
        # that finds a leak but cannot repair it (``_break_equation_duplication``
        # reporting no change) makes no change either, so the loop reads that
        # as its fixed point and stops. An equation would then leave RPS
        # carrying another equation's explained quantity, silently -- which is
        # how this class of defect stayed invisible in the first place.
        #
        # Rescan only when the LAST pass still saw a leak: a pass that found
        # none has already proved the invariant, so the normal exit adds no
        # scans at all. Diagnostic, not fatal -- a leak is a quality defect
        # (unlike a duplicate term, which crashes the next operator), and
        # aborting a whole evolutionary run over one offspring would be worse
        # than the leak. It reaches a user who asked for search warnings
        # (``init_verbose(show_warnings=True)``) and is always measurable via
        # ``EPDE_LOOP_STATS``.
        if leaks_seen:
            unrepaired = []
            for eq_i in equations:
                for eq_j in equations:
                    if eq_i is eq_j:
                        continue
                    sig = _target_term_in_other_equation(eq_i, eq_j)
                    if sig is None:
                        continue
                    carrier = next((t for t in eq_j.structure
                                    if t.factors_labels == sig), None)
                    what = carrier.name if carrier is not None else str(sorted(sig))
                    unrepaired.append(f'{eq_j.main_var_to_explain} carries {what} '
                                      f'(reserved by {eq_i.main_var_to_explain})')
            if unrepaired:
                _loop_stats.record('SoEqRPS.unrepaired_leaks_at_exit',
                                   len(unrepaired), max_passes)
                warnings.warn(
                    'SoEqRightPartSelector.apply: target-term uniqueness not '
                    f'reached after {passes_used} of {max_passes} passes; '
                    f'{len(unrepaired)} leak(s) left: ' + '; '.join(unrepaired[:4]))

    def use_default_tags(self):
        self._tags = {'right part selection', 'chromosome level',
                      'contains suboperators', 'inplace'}
