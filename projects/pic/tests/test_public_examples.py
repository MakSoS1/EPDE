"""Published examples are immutable scientific records, not visitor sessions."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest
from support.discovery_lab import job_queue as queue
from support.discovery_lab.portable import input_key

APP=Path(__file__).resolve().parents[1]/'app'


def library():
    module=importlib.import_module('support.discovery_lab.public_examples')
    return module.load_library()


def test_packaged_public_groups_have_four_real_methods_and_verified_inputs():
    groups=library()
    assert len(groups)>=2
    for group in groups:
        runs=group['runs']
        assert {r['record']['method'] for r in runs}=={'epde','pdecomp','deepmod','discover'}
        assert len({input_key(r['record']) for r in runs})==1
        assert input_key(runs[0]['record']) is not None
        for run in runs:
            r=run['record']
            assert r['problem']['source']=='synthetic'
            assert r['provenance']['workflow_run']
            assert 'data_file' not in r
            assert run['sha256']==hashlib.sha256(run['raw']).hexdigest()


def test_public_loader_returns_independent_copies():
    first=library();first[0]['runs'][0]['record']['selected']=['modified']
    assert library()[0]['runs'][0]['record']['selected']!=['modified']


def test_public_loader_rejects_untrusted_paths_and_modified_bytes():
    module=importlib.import_module('support.discovery_lab.public_examples')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)
        manifest={'schema':'epde-public-results/1','groups':[{'id':'bad','runs':[
            {'file':'../private.json','sha256':'0'*64}]}]}
        (p/'index.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError):module.load_library(p)


def test_empty_private_session_opens_public_examples_without_creating_jobs():
    with tempfile.TemporaryDirectory() as td,patch.object(queue,'ROOT',Path(td)),\
            patch.object(queue,'ensure_manager',side_effect=AssertionError('must not execute')):
        app=AppTest.from_file(str(APP/'Home.py'),default_timeout=60).run()
        app.switch_page('views/11_Discovery_results.py').run()
        assert not app.exception
        assert any('Public results' in str(x.value) for x in app.get('button_group'))
        assert any(w.label=='Public example' for w in app.selectbox)
        assert not any(b.label=='Delete this result' for b in app.button)
        assert not list(Path(td).glob('owners/*/jobs/*/request.json'))


def test_public_compare_selects_all_four_methods_and_explains_unscored_results():
    with tempfile.TemporaryDirectory() as td,patch.object(queue,'ROOT',Path(td)):
        app=AppTest.from_file(str(APP/'Home.py'),default_timeout=60).run()
        app.switch_page('views/12_Discovery_compare.py').run()
        assert not app.exception
        picked=next(x for x in app.multiselect if x.label=='Select runs')
        assert len(picked.value)==4
        assert any(t.label=='Insights' for t in app.tabs)
        assert any('curated' in c.value.lower() for c in app.caption)
