"""Helpers of the Streamlit app: paths, jobs, equation formatting, result views.

The app is a thin shell over ``epde_bench``: every search it starts goes through
``bench.py`` (or ``custom_run.py`` for uploaded data) in a separate process and
writes the same JSON record as the command line and the notebooks.
"""
