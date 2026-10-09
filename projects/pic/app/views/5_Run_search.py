import time

import streamlit as st

from support import jobs
from support.common import cpu_count, dataset_label, dataset_names, page, read_json
from support.options import search_options
from support.viewer import render_record

page('Run a search', '🚀')
st.caption('Choose a record and press Start. The search runs in the background and its result is saved; '
           'the stored settings of the record are used unless you change them under More options.')

names = dataset_names()
c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment='bottom')
name = c1.selectbox('record', names, index=names.index('ode'), format_func=dataset_label)
noise = c2.number_input('noise, %', 0.0, 50.0, 0.0, 0.5)
start = c3.empty()
variant, seed, overrides, problem = search_options(name)
if problem is not None:
    if problem.truth:
        from support.equations import system_latex
        st.markdown('**Known law of this record** — the search does not see it; it is used only to score the result')
        st.latex(system_latex(problem.truth, problem.axis_names))
    else:
        st.caption('The governing law of this record is not known, so the result will not be scored.')
if start.button('Start', type='primary', width='stretch', disabled=problem is None):
    job = jobs.start_run(name, variant, float(noise), seed, overrides)
    st.session_state['run_job'] = job['id']
    st.toast(f"started: {job['title']}")

from support.heldout_view import heldout_example

heldout_example()

all_jobs = jobs.jobs(('run', 'custom'))
if not all_jobs:
    st.stop()

running = [j for j in all_jobs if jobs.status(j) == 'running']
if len(running) >= cpu_count():
    st.warning(f'{len(running)} searches are running on {cpu_count()} CPU threads; new ones will be slow.')

st.divider()
labels = {j['id']: f"{j['title']} — {jobs.status(j)}" for j in all_jobs}
ids = list(labels)
current = st.session_state.get('run_job', ids[0])
job_id = st.selectbox('search', ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
job = next(j for j in all_jobs if j['id'] == job_id)
state = jobs.status(job)

if state == 'running':
    a, b = st.columns([4, 1], vertical_alignment='center')
    a.info(f"Running for {time.time() - job['started']:.0f} s. The page refreshes by itself.")
    if b.button('Stop', width='stretch'):
        jobs.stop(job)
        st.rerun()
    with st.expander('log'):
        st.code(jobs.log_tail(job), language=None)
    time.sleep(5)
    st.rerun()
elif state == 'stopped':
    st.error('The search was stopped or ended without a result.')
    with st.expander('log'):
        st.code(jobs.log_tail(job, 60), language=None)
else:
    record = read_json(job['out'])
    if record:
        render_record(record, key=job_id)
    else:
        st.error('The search failed before writing a result.')
        with st.expander('log'):
            st.code(jobs.log_tail(job, 60), language=None)
