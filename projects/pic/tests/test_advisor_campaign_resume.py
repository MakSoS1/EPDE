"""A resumed sweep keeps its original experiments, despite float-rounding IDs."""
import importlib.util
import json
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('advisor_campaign',Path(__file__).resolve().parents[1]/'scripts/advisor_campaign.py')
campaign=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(campaign)


def test_resume_uses_frozen_request_when_descriptor_rounding_changes_id(tmp_path):
    old={'method':'epde','dataset':'nir1_heat','noise':0.,'seed':0,'policy':'rule',
         'run_id':'old','vector':[1.,2.],'overrides':{'x':1},'semantic_signature':'same'}
    folder=tmp_path/'runs/old';folder.mkdir(parents=True)
    (folder/'request.json').write_text(json.dumps(old))
    new=dict(old,run_id='new',vector=[1.,2.+1e-14])
    assert campaign.reuse_frozen_requests([new],tmp_path)==[old]
    new['overrides']={'x':2}
    with pytest.raises(ValueError):campaign.reuse_frozen_requests([new],tmp_path)
