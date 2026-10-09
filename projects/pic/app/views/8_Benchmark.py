import time

import pandas as pd
import streamlit as st

from support import jobs
from support.common import KIND_NAMES, RESULTS_DIR, cpu_count, dataset_label, dataset_names, page, variant_label, variants

page('Benchmark', '🏁')
st.caption('A campaign runs every combination of records, methods, noise levels and seeds and compares '
           'how often each method finds the known law.')

tab_read, tab_start = st.tabs(['Results', 'Start a campaign'])

with tab_start:
    with st.form('campaign'):
        datasets = st.multiselect('records', dataset_names(with_truth=True), default=['ode', 'vdp', 'burgers'],
                                  format_func=dataset_label)
        c1, c2 = st.columns(2)
        var = c1.multiselect('methods', variants(), default=['default', 'pysindy'], format_func=variant_label)
        noise = c2.multiselect('noise, %', [0, 0.5, 1, 2, 5, 10], default=[0, 1])
        with st.expander('More options'):
            name = st.text_input('name', f"campaign_{time.strftime('%m%d_%H%M')}")
            c1, c2 = st.columns(2)
            seeds = c1.text_input('seeds', '0-2', help='a range such as 0-2 or a list such as 0,3,5')
            workers = c2.slider('parallel processes', 1, cpu_count(), max(1, cpu_count() - 2))
        go = st.form_submit_button('Start', type='primary')
    if go and datasets and var and noise:
        if (RESULTS_DIR / name).exists():
            st.error('A campaign with this name exists; choose another name (results are never overwritten).')
        else:
            jobs.start_campaign(name, datasets, var, noise, seeds, workers)
            st.success(f'started {name}')
    seen = set()
    for job in jobs.jobs('campaign'):
        if job['campaign'] in seen:
            continue
        seen.add(job['campaign'])
        state = jobs.status(job)
        done, planned = jobs.campaign_progress(job['campaign'])
        st.write(f"**{job['campaign']}** — {state}")
        st.progress(done / planned if planned else 0.0, text=f'{done} of {planned} runs')
        actions = st.columns(3)
        if state == 'running':
            if actions[0].button('Stop', key=f"stop_{job['id']}"):
                jobs.stop(job)
                st.rerun()
        else:
            if actions[0].button('Resume', key=f"resume_{job['id']}"):
                jobs.resume_campaign(job)
                st.rerun()
            if actions[1].button('Retry failed runs', key=f"retry_{job['id']}"):
                jobs.resume_campaign(job, retry_errors=True)
                st.rerun()
        if state == 'running':
            with st.expander('log'):
                st.code(jobs.log_tail(job), language=None)

with tab_read:
    campaigns = sorted([p for p in RESULTS_DIR.iterdir() if (p / 'campaign.json').exists()],
                       key=lambda p: p.stat().st_mtime, reverse=True) if RESULTS_DIR.exists() else []
    if not campaigns:
        st.info('No campaigns yet. Start one in the next tab or from the command line.')
        st.stop()
    folder = st.selectbox('campaign', campaigns, format_func=lambda p: p.name)
    done, planned = jobs.campaign_progress(folder.name)
    st.progress(done / planned if planned else 0.0, text=f'{done} of {planned} runs finished')
    report = folder / 'report'
    if (done and not (report / 'success_front.csv').exists()) or st.button('Refresh the report'):
        from epde_bench.report import make_report
        with st.spinner('building the report…'):
            make_report(folder)
    if (report / 'success_front.csv').exists():
        long = pd.read_csv(report / 'success_front.csv')
        noise_levels = sorted(long.noise.unique())
        level = st.select_slider('noise, %', noise_levels) if len(noise_levels) > 1 else noise_levels[0]
        metric = st.radio('count a run as a success when', ['success_selected', 'success_front'], horizontal=True,
                          format_func={'success_front': 'the law is among the candidates',
                                       'success_selected': 'the selected equation is the law'}.get)
        part = long[long.noise == level].assign(dataset=lambda d: d.dataset.map(dataset_label),
                                                variant=lambda d: d.variant.map(variant_label))
        table = part.pivot_table(index='dataset', columns='variant', values=metric)
        st.dataframe(table.style.format('{:.0%}').background_gradient(cmap='Greens', vmin=0, vmax=1),
                     width='stretch')
        times = part.pivot_table(index='dataset', columns='variant', values='median_fit_s')
        st.markdown('**Median search time, s**')
        st.bar_chart(times)
        with st.expander('ranking, figures and the full report'):
            if (report / 'ranking.csv').exists():
                st.markdown('**Best variant per problem class**')
                rank = pd.read_csv(report / 'ranking.csv')
                rank['variant'] = rank['variant'].map(variant_label)
                rank['kind'] = rank['kind'].map(KIND_NAMES).fillna(rank['kind'])
                st.dataframe(rank.rename(columns={'kind': 'class', 'variant': 'method', 'success_front': 'truth on the front',
                                                  'success_selected': 'compromise pick correct',
                                                  'median_fit_s': 'median time, s'}),
                             hide_index=True, width='stretch')
            for png in sorted(report.glob('fig_*.png')):
                st.image(str(png), caption=png.stem)
            if (report / 'summary.md').exists():
                st.markdown((report / 'summary.md').read_text(encoding='utf-8'))
    else:
        st.info('No finished runs yet.')
