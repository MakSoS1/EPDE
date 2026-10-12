"""Small executable public-interface examples beside the existing page controls.

Each example includes its own setup; code is displayed, never run on page load.
"""
from dataclasses import dataclass
import streamlit as st


@dataclass(frozen=True)
class Example:
    title: str
    notes: str
    code: str


# Shared text keeps the displayed examples and their execution tests identical.
OSCILLATOR = '''import numpy as np
from epde import EpdeSearch

t = np.linspace(0, 6, 121)
decay = np.exp(-0.25 * t)
u = decay * np.sin(t)
# One stack per variable; axis-major columns: u_t, u_tt.
derivs = [np.column_stack((
    decay * (np.cos(t) - 0.25 * np.sin(t)),
    decay * (-0.9375 * np.sin(t) - 0.5 * np.cos(t))))]
search = EpdeSearch(config={
    'domain': {'boundary_width': 5},
    'preprocessing': {'max_deriv_order': [2]},
    'search_space': {'data_fun_pow': 2, 'deriv_fun_pow': 1,
                     'equation_terms_max_number': 4,
                     'equation_factors_max_number': 1},
    'evolution': {'population_size': 4, 'training_epochs': 3,
                  'neighbors_number': 2},
})
domain_id, domain = search.createDomain([t], ID=0)
trajectory_id, trajectory = search.createTrajectory(
    {'u': u}, domain, cache_id=0, derivs=derivs)
'''

EXAMPLES = {
    'settings': Example('Configure EpdeSearch', '''**Input:** a grouped dictionary or a JSON/YAML path.
**Output:** a search object and its resolved `search.config`; no equations yet.
Built-in defaults are overridden by `config`, then by explicit constructor keywords.
The seven groups are domain, preprocessing, search_space, objectives, solver, evolution
and runtime. Benchmark `load_config` also includes loader/method settings; pass its
resolved search group to `EpdeSearch`, rather than the entire benchmark configuration.
EPDE settings and caches are process globals: use separate processes for independent runs.''', '''from dataclasses import asdict
from epde import EpdeSearch

search = EpdeSearch(config={'evolution': {'population_size': 4}},
                    training_epochs=3)
print(asdict(search.config.evolution))
search.close()
'''),
    'domain': Example('Build a domain and trajectory', '''**Input:** matching coordinate and field arrays, with time first.
**Output:** `(ID, Domain)` and `(ID, Trajectory)` pairs. A domain specifies the grid
and excluded boundary. A trajectory attaches variables, optional supplied derivatives
and a preprocessing pipeline; it is not a fitted equation. These exact derivatives
belong to this exact field. Omit `derivs=derivs` to calculate derivatives from the field.
For a PDE, use full `np.meshgrid(..., indexing='ij')` coordinate tensors and fields
with the same shape. Derivative columns contain each axis's orders in axis-major order,
with shape `(field.size, sum(max_deriv_order))` per variable; mixed derivatives are
not included in this stack.''', OSCILLATOR + '''print(domain_id, trajectory_id, u.shape)
search.close()
'''),
    'preprocessing': Example('Run preprocessing without discovery', '''**Input:** a sampled field, coordinate tensors and derivative orders per axis.
**Output:** the processed field and derivative stack. Here the two columns are `u_t`
and `u_tt`. This native pipeline performs no evolutionary search. `FD`, `poly`,
`spectral` and `ANN` select different derivative assumptions; each has its own setup
options. Smoothing can change the field itself, so use the returned field alongside
its derivatives. In the benchmark layer, `prepare_data(problem, data, resolved_cfg)`
returns two dictionaries keyed by variable and validates supplied stacks.''', '''import numpy as np
from epde.preprocessing.preprocessor import ConcretePrepBuilder
from epde.preprocessing.preprocessor_setups import PreprocessorSetup

t = np.linspace(0, 6, 121)
u = np.exp(-0.25 * t) * np.sin(t)
setup = PreprocessorSetup()
setup.builder = ConcretePrepBuilder()
setup.build_FD_preprocessing()
pipeline = setup.builder.prep_pipeline
field, derivatives = pipeline.run(u.copy(), grid=[t], max_order=[2])
print(field.shape, derivatives.shape)  # (121,), (121, 2)
'''),
    'build': Example('Prepare a benchmark record with build_search', '''**Input:** a loaded `Problem`, resolved search configuration and optionally replacement
fields. **Output:** `(search, trajectory, extra_token_families)`, ready to pass to
`fit`. This project helper wraps `EpdeSearch`, domain/trajectory creation, validation
and the record's supplied derivative/forcing families; it does not discover equations.
`load_config` merges the base protocol, record settings, variant and overrides;
`resolve_for_problem` expands values such as automatic boundary widths and orders.
Do not reuse supplied derivatives when replacing a field with noisy measurements.
`discover` additionally calls `fit` and returns search, equation texts, objectives and
fit seconds. `run_one` also handles seeded noise, scoring and run-record provenance;
the application's Start control launches that workflow in its own process.''', '''from epde_bench import datasets, load_config, resolve_for_problem
from epde_bench.runner import build_search

problem = datasets.load('vdp')
cfg = resolve_for_problem(load_config('vdp'), problem)
search, trajectory, families = build_search(problem, cfg)
print(problem.variables, len(families), type(trajectory).__name__)
# To discover: search.fit(data=[trajectory], additional_tokens=families)
search.close()
'''),
    'pool': Example('Build the token pool explicitly', '''**Input:** trajectories, derivative orders, powers and optional token families.
**Output:** `search.pool`; `create_pool` prepares derivative/variable factors but does
not evolve equations. Pool structure is shared across trajectories, which differ in
evaluation. Thus `max_deriv_order`, `data_fun_pow`, `deriv_fun_pow` and declarative
`additional_tokens` belong to `create_pool` or `fit`, rather than `createTrajectory`.
`cached_token_tensors` on a trajectory is a custom-family tensor upload map, not a
list of token families. Add measured forcing or other physically justified families
before discovery; more epochs cannot supply a missing factor.''', OSCILLATOR + '''from epde import GridTokens

families = [GridTokens(['x'], dimensionality=0, max_power=1)]
search.create_pool(data=[trajectory], additional_tokens=families,
                   max_deriv_order=[2], data_fun_pow=2, deriv_fun_pow=1)
print(type(search.pool).__name__)
search.close()
'''),
    'fit': Example('Discover equations with fit', '''**Input:** prepared trajectories and optional token families; the resolved settings
supply size limits, objectives and evolution budget. **Output:** fitted search state;
`fit` returns `None`. `search.equations(only_print=False, num=1)` returns the first
Pareto level(s) in multiobjective mode. The benchmark `pareto_front` helper converts
that state into equation text and objective vectors.
`EpdeSearch.fit` includes pool preparation, candidate coefficient fitting and evolution;
it is different from fitting coefficients to a known structure on Signal check.
This tiny exact oscillator is an API exercise, not a reliability benchmark.''', OSCILLATOR + '''from epde_bench.runner import pareto_front

search.fit(data=[trajectory])
texts, objectives = pareto_front(search)
print(texts, objectives)
search.close()
'''),
    'optimizer': Example('Run the optimizer directly', '''**Input:** an existing token pool, population instructions and strategy director.
**Output:** the optimizer's Pareto population and observed epoch snapshots. Direct use
bypasses `EpdeSearch.fit` orchestration. EPDE's existing strategy still fits and scores
each candidate. This example uses the same objectives and operators as the short
optimizer button below. Callback epochs are zero-based; snapshots contain serialized
`text_form` and `obj_fun`. Returning `False` keeps the search running. The final front
may be unchanged across epochs; the separate column minima need not describe one
candidate. This is the multiobjective optimizer; single-objective searches use a
different optimizer and population representation.''', OSCILLATOR + '''import copy
from epde.optimizers.moeadd.moeadd import MOEADDOptimizer
from epde.operators.common.objectives import ideal_point

search.create_pool(data=[trajectory])
params = dict(search.optimizer_init_params)
params['population_instruct'] = {
    'pool': search.pool, 'terms_number': 4, 'max_factors_in_term': 1,
    'sparsity_interval': (1., 1.), 'second_objective': 'instability'}
params['best_sol_vals'] = ideal_point(('discrepancy', 'instability'))
optimizer = MOEADDOptimizer(**params)
optimizer.set_strategy(search.director)
history = []
def observe_front(snapshot, epoch_idx):
    history.append({'epoch': epoch_idx + 1, 'front': copy.deepcopy(snapshot)})
    return False
optimizer.optimize(epochs=3, early_stopping_callback=observe_front)
for entry in history:
    print(entry['epoch'], [(s['text_form'], s['obj_fun']) for s in entry['front']])
search.close()
'''),
}

PAGE_EXAMPLES = {
    'EPDE explorer': ('settings',),
    'Data sets': ('domain', 'build'),
    'Signal check': ('preprocessing',),
    'Derivatives': ('preprocessing', 'domain'),
    'Run a search': ('settings', 'build', 'pool', 'fit'),
    'Your data': ('domain', 'pool', 'fit'),
    'Results': ('fit',),
    'Benchmark': ('settings', 'build'),
}
STAGE_EXAMPLES = {
    'settings': 'settings', 'domain': 'domain', 'trajectory': 'domain',
    'pool': 'pool', 'population': 'optimizer', 'fit': 'fit',
    'sparsity': 'optimizer', 'objectives': 'optimizer', 'evolution': 'optimizer',
    'front': 'optimizer', 'solve': 'fit',
}


def show_examples(keys):
    with st.expander('Python interfaces'):
        st.caption('Run each example in a fresh Python process with the project dependencies '
                   'installed. Examples using epde_bench require projects/pic on PYTHONPATH '
                   '(or its editable installation). Each block is self-contained; page load '
                   'only displays code. Close the search after inspecting its results.')
        for key in keys:
            example = EXAMPLES[key]
            st.markdown(f'**{example.title}**')
            st.markdown(example.notes)
            st.code(example.code, language='python')


def page_interfaces(title):
    if title in PAGE_EXAMPLES:
        show_examples(PAGE_EXAMPLES[title])


def stage_interfaces(stage):
    show_examples((STAGE_EXAMPLES[stage],))
