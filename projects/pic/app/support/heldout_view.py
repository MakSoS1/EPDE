"""A small chronological validation example beside the existing search UI."""
import json
import os
import subprocess
import sys
import uuid

import streamlit as st

from .common import APP_DIR, PIC_DIR, REPO_ROOT, APP_RESULTS, read_json


def heldout_example():
    with st.expander('Training on 70%, forecasting the next 30%'):
        st.markdown(
            'This separate oscillator example splits **raw observations before differentiation**. '
            'EPDE sees the first 224 samples only. Candidate selection uses training residuals; '
            'the equation is then frozen and integrated over the next 96 samples. '
            'The initial state and velocity come from training observations only. '
            'Future observations are used to score the finished forecast, never to tune it.'
        )
        st.code(
            "from epde_bench.heldout import run_heldout\n"
            "record = run_heldout('ode', train_fraction=0.7, seed=0,\n"
            "                     output_dir='results/my_heldout_example')\n"
            "print(record['selected'])\n"
            "print(record['test']['metrics'])", language='python'
        )
        destination = PIC_DIR / 'experiments' / 'heldout_2026-10-09'
        origin = 'Published execution from 9 October 2026'
        if st.button('Run training-only example', key='heldout_start'):
            destination = APP_RESULTS / 'heldout' / uuid.uuid4().hex
            destination.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                               MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
            with st.spinner('Discovering from training observations and forecasting…'):
                try:
                    with (destination / 'execution.log').open('w') as log:
                        result = subprocess.run(
                            [sys.executable, str(APP_DIR / 'support' / 'heldout_run.py'), str(destination)],
                            cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                            timeout=900, check=False)
                    if result.returncode and not (destination / 'record.json').exists():
                        st.error('The example failed before saving a record.')
                        st.code((destination / 'execution.log').read_text()[-6000:])
                except subprocess.TimeoutExpired:
                    st.error('The training-only example exceeded its 15-minute limit.')
            st.session_state['heldout_destination'] = str(destination)
        if 'heldout_destination' in st.session_state:
            from pathlib import Path
            destination = Path(st.session_state['heldout_destination'])
            origin = 'Executed in this application session'
        record = read_json(destination / 'record.json')
        if not record:
            st.caption('Run the example to save its equation, forecast and scores.')
            return
        st.caption(origin)
        st.write('Execution status:', record['status'])
        st.write('Frozen equation selected using training observations:')
        from .equations import system_latex
        if record.get('selected'):
            st.latex(system_latex(record['selected']))
        st.json({'training': record['train'], 'held_out': record['test']}, expanded=False)
        if (destination / 'trajectory.png').exists():
            st.image(str(destination / 'trajectory.png'))
        st.caption('Assess forecast accuracy and the learned terms separately: a good forecast '
                   'can coexist with extra terms. This validates one scalar ODE case; '
                   'other records require their own chronological evaluation.')
        st.download_button('Download validation record', json.dumps(record, indent=2),
                           'heldout-record.json', 'application/json', key='heldout_download')
