"""Reading notes beside the existing controls, with no alternate interface."""
import streamlit as st

LESSONS = {
    'EPDE explorer': '''
**Read, experiment, inspect.** Start with a synthetic ordinary equation on the Data sets page.
Look at its signal and known law, compare derivatives, then run a search with the shipped settings.
The resulting equations are saved, so you can inspect them again on Results.

**Where computation runs.** This application runs Python on the machine hosting Streamlit.
For a local launch that is your computer; for a hosted deployment it is the hosting server,
not the computer displaying the browser. Keep the application terminal open for a local search;
closing a browser tab does not stop the separate search process. Use Stop on the search page.

**A first exercise.** Keep the same record and seed. Change only the noise level, then compare
term contributions, the Pareto front and the reconstructed trajectory. This isolates one effect
instead of changing several choices at once. The benchmark page repeats the same protocol across
records and seeds.
''',
    'Data sets': '''
**What is a record?** A record contains sampled variables and their coordinates. For an ordinary
equation the coordinate is time. A field also has spatial coordinates; time comes first.
Several variables may share the same grid and form a jointly discovered system.

**Known and unknown laws.** Synthetic records provide a reference equation for structural scoring.
Some records accept several equivalent laws. Measured data usually have no verified reference;
a plausible equation then needs physical interpretation and independent validation.

**Try it.** Select an ordinary equation, inspect its signal and law, then select a system and
compare the variables. Missing external files are reported explicitly. Viewing a higher-dimensional
field is a slice of the data and is not evidence that an equation has been discovered.
''',
    'Signal check': '''
**What this check answers.** Before searching, ask whether the expected terms have an observable
contribution under the selected differentiation and noise. The known structure is fitted directly;
this is a diagnostic using the reference law, not a blind equation discovery.

**Read the two scores together.** A high overall R² can coexist with a negligible diffusion term.
The loss when a term is removed measures its incremental contribution in this regression.
Correlated terms and the chosen preprocessing can make a contribution look weak. This does not
prove that every identification method must fail.

**Try it.** Compare clean data with progressively stronger noise. Keep the seed fixed while
changing derivatives, then change the seed to check that the conclusion survives another noise
realisation. Do not compare percentages from different scoring protocols as if they were identical.
''',
    'Derivatives': '''
**Why derivatives matter.** The search compares terms containing derivatives of measured fields.
Differentiation amplifies noise, especially at high orders, so an apparently small measurement
error can change which structures fit well.

**Choose an assumption.** Finite differences are local; polynomial fits average a neighbourhood;
spectral differentiation benefits from smooth compatible boundaries; a neural approximation
introduces fitting and runtime choices. No method is uniformly best. Compare them on the same
record, noise level and derivative order.

**Try it.** Begin with clean data and a first derivative, then raise the order or add noise.
Inspect edge behaviour and smoothing bias rather than choosing solely by a visually smooth curve.
When supplied derivatives exist, they belong to that recorded field and cannot be reused for an
independently perturbed field.
''',
    'Run a search': '''
**What Start does.** It launches one independent Python process with the selected record,
method, noise and seed. It saves the effective settings, environment, equations and scores as a
JSON record. The result is reproducible within the recorded software and data protocol.

**Reading the front.** Each candidate trades discrepancy against stability or size. Several
candidates can share the same structure. The highlighted compromise is a selection rule, not a
guarantee of the correct law. Invalid objective vectors are excluded from that selection.

**Options.** By default a search uses the settings stored for the record. *More options* changes
the method, seed, derivative method, search effort and equation size; the *Expert* tab exposes
the objectives, sparsity and exact budget. An absent forcing function cannot be recovered by
running longer: the hypothesis space is set by powers, derivative orders and token families.

**Try it.** Run a small ordinary-equation record first. Inspect its reference law, term set and
coefficients, then the integrated trajectory. A short fit or a high in-sample R² alone is insufficient
validation. A stopped or failed process is shown explicitly, with its log.
''',
    'Your data': '''
**Prepare a finite grid.** Coordinates must be finite, strictly increasing and match every
variable's shape. Use clear variable names. A time series may contain multiple jointly discovered
variables. A field requires explicit time and spatial coordinates, not only a flattened list.

**File formats.** A CSV table with a header: the first column is time, every other column is
a variable. A CSV or NPY matrix without a header: one field, time along the rows and space along
the columns; the time and space ranges are entered on the page. An NPZ archive for full control:
coordinate arrays named axis_0 (time) and axis_1 (space, optional), names in axis_names, and one
array per variable named var_ followed by the variable name.

**Choosing a model.** Begin with plausible derivative orders and a small term library. Include
known measured forcing fields where available. Increasing the library without a physical reason
can create many equally accurate explanations.

**After the search.** Your uploaded record has no known governing law, so structural correctness
is not scored. Inspect reconstruction, units and plausible terms, then evaluate on independently
reserved data. The application saves data and result records on the machine hosting
Streamlit for later inspection. Hosted deployments need persistent storage to retain
these files across restarts.
''',
    'Results': '''
**A result is evidence from a specific run.** Read the seed, noise level, derivative method and
resolved settings before comparing two records. The displayed noise realisation is reproduced
from the recorded seed; it is not replaced with the clean field.

**Agreement versus prediction.** Comparing the two equation sides on sampled data checks a
local residual. Integrating an ordinary equation lets errors accumulate. A failed integration
cannot earn a good score just because its short surviving prefix fits well.

**Try it.** Compare several candidates from the same front. Check whether the known structure
is present and whether the selection rule chose it. Reconstruction uses the fitted record;
an independent forecast requires fitting without the test interval and evaluating that interval
separately. Unknown laws need an external validation argument.
''',
    'Benchmark': '''
**One comparison protocol.** A campaign specifies records, methods, noise levels and seeds.
Each planned run executes independently. Resume keeps completed records; retrying failed runs
repeats only the relevant attempts. A manifest existing on disk does not mean a campaign completed.

**Read the denominator.** Unsupported equation representations are different from failed
searches. Report both. Recovery anywhere on a Pareto front differs from recovery by its chosen
candidate, and per-equation success differs from success of an entire system.

**Try it.** Begin with one small record, one seed and two methods. Inspect their actual equations
before expanding the campaign. Historical notebook tables are saved illustrations until the raw
records are re-scored with the corrected metric; new results retain their own provenance.
''',
    'How EPDE works': '''
**Follow one candidate.** Data and derivatives define a pool of possible factors. Candidate
terms combine those factors into equations. Fitting assigns coefficients and removes weak terms;
objectives score accuracy and stability or size. Evolution changes structures, refits them and
updates the non-dominated candidates.

**Explore the stages.** Select a stage in the existing diagram and follow its Try it link.
The small evolution example below runs the actual optimizer and records epoch snapshots.
A stationary front is a possible outcome and must not be presented as improvement.
''',
}

def reading_notes(title):
    lesson = LESSONS.get(title)
    if lesson:
        with st.expander('About this page'):
            st.markdown(lesson)
