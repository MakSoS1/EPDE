"""Private ZIP imports are visible without a second click."""
import io
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from support.discovery_lab import job_queue as queue
from test_portable_handoff import mock_record, archive
APP=Path(__file__).resolve().parents[1]/'app'
def test_private_import_is_visible_in_the_same_rerun(tmp_path):
    upload=io.BytesIO(archive([mock_record()]))
    with patch.object(queue,'ROOT',tmp_path), patch('streamlit.file_uploader',return_value=upload):
        app=AppTest.from_file(str(APP/'Home.py'),default_timeout=60).run()
        app.session_state['discovery_result_collection']='My results'
        app.switch_page('views/11_Discovery_results.py').run()
        assert not app.exception
        next(b for b in app.button if b.label=='Import these results').click().run()
        assert not app.exception
        assert any(m.label=='Experiments' and m.value=='1' for m in app.metric)
        assert not any('No experiments yet' in n.value for n in app.info)
