# DeepMoD and DISCOVER in the Streamlit demo

Run `projects/pic/app/Home.py`, open **Run a search**, and select **DeepMoD**
and/or **DISCOVER**. Each has independent settings and defaults to **native**.
The optional **fixed** protocol uses PDEComp's shared candidate library.
Results, Compare, seed/noise series and native experiment ZIPs accept the new
methods alongside EPDE and PySINDy. There is no automatic protocol fallback.

| Method | Native execution | Supported native input |
| --- | --- | --- |
| DeepMoD | DeePyMoD NN, autograd derivatives, stock Library1D, sparse estimator and least-squares constraint | Scalar 1D PDE with first-order time target `u_t` |
| DISCOVER | Recurrent controller trained with policy gradients, symbolic operator search, DISCOVER finite differences | Scalar ODE or 1D PDE, target `u_t` or `u_tt` |

These restrictions come from PDEComp's wrappers. DeepMoD's native pipeline
does not accept ODEs, coupled systems, second-order time targets or 2D PDEs.
Fixed DeepMoD supports additional benchmark formulations. DISCOVER supports
neither systems nor 2D PDEs in either protocol.

PDEComp's per-dataset profiles are used for its registered datasets. Other
built-in records and uploaded/generated data use its native defaults, with
the target derivative taken from the declared law or the upload controls.
No reference coefficients are supplied to either method. For fixed runs on
new data, a shared polynomial/Fourier library is built with the chosen target.

Training iterations/sample counts are editable. Short budgets verify that
the method runs; they do not establish recovery accuracy. All hosted jobs
retain the existing queue, cancellation and wall-time limit. Heavy records
are exported for local execution and never loaded by the Streamlit host.
DeepMoD/DISCOVER workers run serially with other searches to limit peak memory.

## Reproducible upstream sources

The wrappers use the same revision already pinned for PySINDy:

- [PDEComp `4abfe4e`](https://github.com/emil817/PDEComp/tree/4abfe4e9eef5263c35ae5672c6ebfb65aabf8365)
- [DeePyMoD `69b53c0`](https://github.com/emil817/DeePyMoD/tree/69b53c056b042c6f9687bff01835b71472c87cb3)
- [DISCOVER `fa1db7b`](https://github.com/emil817/DISCOVER/tree/fa1db7b1c51cd8df27c6e524613f1bdaf23ae579)

Source files are fetched on first use into `projects/pic/.cache/pdecomp/`
and checked against `epde_bench/pdecomp_sources.json`. An existing verified
checkout can be supplied with `PDECOMP_DIR`; missing or modified sources fail
explicitly. The manifest participates in experiment fingerprints.

Each run uses a separate Python process and temporary working directory, so
upstream package names, TensorFlow state and training logs cannot affect the
Streamlit interpreter or another run.

## DISCOVER compatibility

The original TF1.14 environment cannot install on Python 3.12. The app pins
TensorFlow 2.20 and legacy Keras 2.20, and supplies the TF1 graph/session,
RNN-cell and dense-layer APIs inside the DISCOVER subprocess. PDEComp turns
attention off in both protocols; attention is explicitly unavailable in this
compatibility layer. The NumPy `tostring` rename and `collections.Mapping`
move are adapted without modifying checksum-verified upstream files.

The search/training algorithms remain those of DISCOVER, but this is a
TensorFlow compatibility environment, not the original TF1 binary stack;
bitwise equality with the historical environment is not claimed.

DISCOVER's original expression, reward and NMSE are retained. Expressions
that fit the EPDE token grammar receive the usual structural metrics and
solution reconstruction. Other valid native expressions are preserved with
an explicit scoring-unavailable note; they are excluded from success-rate
summaries. No EPDE Pareto scores are invented for either new method.
Equation-balance evaluation uses independent finite differences, rather than
claiming to reuse DeepMoD's learned neural derivatives.

## Checks

With the `cpu` and `app` extras installed:

```powershell
$env:PYTHONPATH = '.;projects/pic;projects/pic/app'
$env:EPDE_TEST_NATIVE_METHODS = '1'
python -m pytest projects/pic/tests/test_native_pdecomp_methods.py -q
```

The native checks actually train a short DeepMoD NN and DISCOVER controller,
including an ODE through the hosted worker. The ordinary tests cover
unsupported inputs, validation, protocol isolation, the wizard, portable
manifests and lossless preservation of native symbolic expressions.
The `streamlit-native-methods.yml` workflow runs these checks on Linux.
