"""Collection preference outlives Streamlit's page-scoped widgets."""
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from support.discovery_lab import job_queue as queue
APP=Path(__file__).resolve().parents[1]/'app'
SOURCE='''
import streamlit as st
from support.discovery_lab.public_results_ui import choose_collection
if st.session_state.get('show_collection',True):
    choose_collection(False)
else:
    st.write('Another page without the collection widget')
'''
def collection(app):
    return next(g for g in app.get('button_group') if g.label=='Result collection')
def test_collection_preference_survives_widget_cleanup():
    app=AppTest.from_string(SOURCE).run()
    assert collection(app).value=='Public results'
    collection(app).set_value('My results').run()
    app.session_state['show_collection']=False
    app.run()
    app.session_state['show_collection']=True
    app.run()
    assert not app.exception
    assert collection(app).value=='My results'
def test_private_collection_survives_results_compare_navigation(tmp_path):
    with patch.object(queue,'ROOT',tmp_path):
        app=AppTest.from_file(str(APP/'Home.py'),default_timeout=60).run()
        app.switch_page('views/12_Discovery_compare.py').run()
        collection(app).set_value('My results').run()
        app.switch_page('views/11_Discovery_results.py').run()
        assert not app.exception
        assert collection(app).value=='My results'
        assert any('No experiments yet' in x.value for x in app.info)
        collection(app).set_value('Public results').run()
        app.switch_page('views/12_Discovery_compare.py').run()
        assert not app.exception
        assert collection(app).value=='Public results'
def test_new_search_can_explicitly_open_private_results():
    app=AppTest.from_string(SOURCE).run()
    app.session_state['discovery_result_collection']='My results'
    app.run()
    assert not app.exception
    assert collection(app).value=='My results'
