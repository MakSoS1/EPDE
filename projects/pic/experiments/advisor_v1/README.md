# CatBoost starting settings — experimental model card

## What is deployed

Three CatBoost regressors predict the selected-equation benchmark loss for each
valid candidate configuration. A separate model estimates runtime on the training
runner and, when needed, a classifier estimates whether shared scoring is available.
The runtime uses checksummed numeric CatBoost tree exports, verified against the
native predictor including float32 split boundaries. No training, LLM requests,
or scientific searches run when the user clicks Suggest settings.

The candidate portfolio contains four configurations per method. The comparator
is **Suggested v1 at the same initial short budget**, not a fully trained Default.
This is a finite-portfolio recommendation, not a global hyperparameter optimum.
Manual settings remain editable. Full local budgets, heavy inputs, fixed protocols,
unsupported profiles and out-of-domain descriptors fall back with an explanation.

## Data and leakage boundaries

The frozen campaign contains eight scalar 1D synthetic PDE datasets, seven equation
families, two added-noise levels (0% and 2%), two seeds, four methods and four policies:
512 requested native scientific runs. Exact requests, outputs, checksums, original
source revisions and workflow IDs are preserved in the campaign artifacts.
`rows.json.gz` contains the complete audit table; unknown scores remain null.

The 18 numerical signal/grid descriptors, declared added noise and configuration
parameters are the model features. Reference equations, dataset names, seed IDs,
family labels and outcome metrics are not features. Actual field values are used
for bounded spectral/roughness descriptors; these are not a calibrated noise estimator.

The reference law supplies the offline supervised target:
`log(1 + selected HD) + 0.25 * log(1 + min(selected RE, 1e6))`.
It does not enter inference. The target is not held-out forecast error.
Every seed, noise level and policy of one family is withheld together.
The two diffusion datasets share one family. See `family-heldout-choices.json`.

## Evaluation and activation

`validation.json` reports every family-held-out choice, scoring coverage, wins/ties/
losses and exact-structure counts. The grouped bootstrap resamples seven family
means; the interval describes the family-balanced mean, while `mean_loss_reduction`
is the separately reported run-weighted mean. Seven families and two seeds are
limited evidence, not a population-wide guarantee.

Automatic model changes require a positive family-balanced gain and positive lower
95% grouped bootstrap bound, at least four scored families, and no reduction in
scoring coverage versus Suggested v1. Methods that fail this gate keep their existing
rules. Availability/missing scores are not converted to zero error or failure labels.
Runtime timeouts are censored observations, so runtime estimates are approximate.

A second baseline selects one fixed policy using training families only. Its held-out
performance is reported separately. An improvement over Suggested v1 is not evidence
of superiority to this fixed baseline or to every other possible configuration.
No hyperparameter/seed search was used to optimize the meta-model on the held-out folds.

## Public results

The read-only gallery uses real rule-baseline outputs on Gaussian diffusion, noisy
transport and noisy advection–diffusion. These are educational examples, not an
unbiased leaderboard and not automatically selected per-method best seeds. Each
example includes all four native methods on one verified input. A separate bounded
offline reconstruction may add plots; its provenance is distinct from the original
search. Original selected equations and reference scores are never rewritten.
No visitor sessions or uploaded private data are harvested or published.

## Reproduce

With all four original or resumed artifacts extracted into method-named folders:

```bash
uv sync --project projects/pic --extra cpu --extra app --locked
uv pip install --python projects/pic/.venv/bin/python catboost==1.2.8 pytest
PYTHONPATH=.:projects/pic:projects/pic/app projects/pic/.venv/bin/python \
  projects/pic/scripts/build_advisor_release.py --campaign-root /tmp/advisor-campaigns
```

The builder rejects incomplete/mixed campaigns and checksum mismatches. Training
requires CatBoost; using the deployed advisor does not. Changing numerical source
or descriptor/selection implementation invalidates the packaged model until rebuilt.

To try a model before it meets the automatic activation gate, enable **Include
experimental CatBoost suggestions** and press **Suggest settings**. This uses the
same trained models and candidate checks for all four methods. The model can
retain the rule-based configuration when alternatives have weak predictions or
exceed resource limits. The experimental option does not change the validation
results or expand the model's supported domain.
