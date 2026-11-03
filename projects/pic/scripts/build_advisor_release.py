#!/usr/bin/env python3
"""Build audited CatBoost assets and read-only examples from genuine sweep artifacts.

Never reads visitor sessions, accepts uploaded models, or fits on page load.
Input directory must contain epde/pdecomp/deepmod/discover campaign folders.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

PIC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PIC),str(PIC/'app'),str(PIC.parents[1])]
from support.discovery_lab.meta_advisor.training import train_bundle, quality_loss
from support.discovery_lab.worker import _sanitize
from advisor_campaign import DATASETS

METHODS=('epde','pdecomp','deepmod','discover')
LABELS={'epde':'EPDE','pdecomp':'PySINDy','deepmod':'DeepMoD','discover':'DISCOVER'}


def encoded(value):
    return json.dumps(_sanitize(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(encoded(value))


def read_campaigns(root, allow_partial=False):
    from epde_bench.identity import _source_hash
    rows=[];records={};provenance={};seen=set()
    for method in METHODS:
        folder=Path(root)/method
        manifest=json.loads((folder/'campaign.json').read_text())
        if manifest['method']!=method or manifest['datasets']!=DATASETS:
            raise ValueError('Campaign definition differs from frozen experiment')
        if manifest['scientific_source_sha256']!=_source_hash():
            raise ValueError('Scientific source has changed since the sweep')
        provenance[method]=manifest
        batch=json.loads((folder/'rows.json').read_text())
        if not allow_partial and len(batch)!=128:
            raise ValueError(f'Incomplete campaign: {method} has {len(batch)} / 128 rows')
        for row in batch:
            key=(method,row['dataset'],row['noise'],row['seed'],row['policy'])
            if key in seen or row['method']!=method:raise ValueError('Duplicate or mixed experimental row')
            seen.add(key)
            if row['family']!=DATASETS.get(row['dataset']) or row['seed'] not in (0,1) or row['noise'] not in (0,2):
                raise ValueError('Unexpected equation family, noise or seed')
            run=folder/'runs'/row['run_id']
            raw=(run/'result.json').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['record_sha256']:
                raise ValueError('Original result checksum mismatch')
            record=json.loads(raw);request=json.loads((run/'request.json').read_text())
            if quality_loss(record)!=row.get('loss') or request['vector']!=row['vector']:
                raise ValueError('Labels or numeric descriptors differ from original artifacts')
            if any(request[k]!=row[k] for k in ('dataset','method','seed','noise','policy','semantic_signature')):
                raise ValueError('Requested scientific configuration does not match row')
            if record.get('status')=='ok':
                if record['identity']['source_sha256']!=_source_hash():
                    raise ValueError('Mixed scientific source fingerprints')
                expected=request['overrides']
                # run_one stores the resolved effective config; preserve it, do not reinterpret labels.
                if not record.get('selected') or not expected:raise ValueError('Missing native output/configuration')
            records[key]=(record,raw)
            if method=='deepmod':
                from support.discovery_lab.meta_advisor.candidates import semantic_signature
                row['original_semantic_signature']=row['semantic_signature']
                row['semantic_signature']=semantic_signature(method,request['settings'])
            rows.append(row)
    counts=Counter((r['method'],r['dataset'],r['noise'],r['seed']) for r in rows)
    if not allow_partial and (len(counts)!=128 or any(v!=4 for v in counts.values())):
        raise ValueError('The frozen four-policy comparisons are incomplete')
    return sorted(rows,key=lambda r:(r['method'],r['run_id'])),records,provenance


# Educational examples deliberately include failures/mismatches, not only winners.
EXAMPLES=(
    ('diffusion','Gaussian diffusion: a recovered law','nir1_h11_gaussian_heat',0.,0,
     'Compare four native methods on the same clean diffusion input. Some recover the reference structure; a completed search is not the same as a correct equation.'),
    ('transport','Transport: short neural training versus search','nir1_advection',2.,1,
     'The same transported field with 2% added noise illustrates different derivative estimators and a short neural budget. Check structure and coefficients separately.'),
    ('noisy-advdiff','Noisy advection–diffusion: limits of a quick run','nir1_advdiff',2.,0,
     'These are intentionally imperfect short-budget results. Missing DISCOVER scores are unavailable, not zero; a plausible fit need not identify the correct law.'),
)


def evaluation(record, timeout=25):
    """Bound reconstruction separately; retain the original scientific fit verbatim."""
    with tempfile.TemporaryDirectory() as td:
        source=Path(td)/'record.json';dest=Path(td)/'evaluation.json'
        save(source,record)
        code=("import json,sys;from support.discovery_lab.worker import evaluate,_sanitize;"
              "r=json.load(open(sys.argv[1]));v=evaluate(r);"
              "json.dump(_sanitize(v),open(sys.argv[2],'w'),allow_nan=False)")
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(PIC.parents[1],PIC,PIC/'app'))))
        with (Path(td)/'evaluation.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-c',code,str(source),str(dest)],env=env,
                                   stdout=log,stderr=log,start_new_session=True)
            try:child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
                return {'solution_note':'Offline reconstruction reached its 25-second limit; the original discovered equation and reference scores remain unchanged.'}
        return json.loads(dest.read_text()) if dest.exists() else {'solution_note':'Offline solution reconstruction was unavailable; original search results are retained.'}


def build_public(records, evaluate=True):
    directory=PIC/'public_results';directory.mkdir(exist_ok=True)
    groups=[]
    for slug,title,dataset,noise,seed,summary in EXAMPLES:
        runs=[]
        for method in METHODS:
            original,raw=records[(method,dataset,noise,seed,'rule')]
            record=deepcopy(original)
            if evaluate:
                record['evaluation']=evaluation(record)
                record['evaluation_provenance']={
                    'kind':'offline in-sample reconstruction of the saved equation, not a new search or held-out forecast',
                    'source_commit':os.environ.get('GITHUB_SHA','local'),
                    'workflow_run':os.environ.get('GITHUB_RUN_ID','local'),
                    'versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','sympy')},
                    'original_result_sha256':hashlib.sha256(raw).hexdigest()}
            record['public_example']={'policy':'rule','original_result_sha256':hashlib.sha256(raw).hexdigest(),
                                     'selection':'educational baseline; no per-method best-seed selection'}
            name=f'{slug}-{method}.json';content=encoded(record)
            (directory/name).write_bytes(content)
            runs.append({'file':name,'sha256':hashlib.sha256(content).hexdigest(),
                         'label':LABELS[method]+' · Suggested v1'})
        groups.append({'id':slug,'title':title,'summary':summary,
                       'selection_note':'Curated educational example, not an unbiased leaderboard. All four runs use their initial Suggested v1 settings on the same noise realization; no visitor data are published.',
                       'runs':runs})
    save(directory/'index.json',{'schema':'epde-public-results/1','groups':groups})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root',required=True)
    parser.add_argument('--allow-partial',action='store_true',help='Development only: disable all learned changes')
    parser.add_argument('--no-evaluate',action='store_true',help='Do not add offline reconstruction plots')
    args=parser.parse_args()
    rows,records,campaigns=read_campaigns(args.campaign_root,args.allow_partial)
    audit=PIC/'experiments/advisor_v1';audit.mkdir(parents=True,exist_ok=True)
    provenance={'campaigns':campaigns,'runs':len(rows),'complete':not args.allow_partial,
                'training_library':'CatBoost '+importlib.metadata.version('catboost'),
                'trained_at':datetime.now(timezone.utc).isoformat(),
                'scope':'eight scalar 1D synthetic PDEs / seven held-out families / 0 and 2 percent added noise / two seeds / four policies per method'}
    bundle,pairs=train_bundle(rows,audit,provenance)
    if args.allow_partial:
        for model in bundle['methods'].values():
            model['enabled']=False;model['report']['enabled']=False
            model['report']['reason']='Incomplete development campaign; automatic changes disabled.'
    asset=PIC/'app/support/discovery_lab/meta_advisor/assets';asset.mkdir(parents=True,exist_ok=True)
    compressed=gzip.compress(encoded(bundle),mtime=0)
    (asset/'model.json.gz').write_bytes(compressed)
    save(asset/'manifest.json',{'schema':bundle['schema'],'sha256':hashlib.sha256(compressed).hexdigest(),
                               'scientific_source_sha256':bundle['scientific_source_sha256'],
                               'implementation_sha256':bundle['implementation_sha256']})
    (audit/'rows.json.gz').write_bytes(gzip.compress(encoded(rows),mtime=0))
    archived=[{'key':list(key),'record_json':raw.decode('utf-8'),
               'sha256':hashlib.sha256(raw).hexdigest()}
              for key,(_,raw) in sorted(records.items())]
    (audit/'native-results.json.gz').write_bytes(gzip.compress(encoded(archived),mtime=0))
    save(audit/'validation.json',{'reports':bundle['reports'],'provenance':provenance,
                                 'metric':bundle['target'],'baseline':bundle['baseline']})
    save(audit/'family-heldout-choices.json',pairs)
    build_public(records,not args.no_evaluate)
    print(json.dumps(bundle['reports'],indent=2))


if __name__=='__main__':main()
