#!/usr/bin/env python3
"""Reproducible native-method sweeps for CatBoost; never executes on page load.

Each scientific fit has a fresh process and bounded time. Failed/unscorable runs
remain in the audit table. Only allow-listed synthetic builtin inputs are used.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PIC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PIC),str(PIC/'app'),str(PIC.parents[1])]
from support.discovery_lab.meta_advisor.candidates import portfolio, vector, changes_for, semantic_signature
from support.discovery_lab.meta_advisor.features import describe_problem
from support.discovery_lab.starting_settings import summarize_input

# Frozen before any outcomes; closely related heat data stay in the same fold.
DATASETS={
    'nir1_heat':'diffusion',
    'nir1_h11_gaussian_heat':'diffusion',
    'nir1_advection':'transport',
    'nir1_advdiff':'advection_diffusion',
    'nir1_reactiondiff':'reaction_diffusion',
    'nir1_h14_cole_hopf':'burgers',
    'nir1_h11_airy':'dispersion',
    'nir1_h11_logistic_field':'local_reaction',
}
VARIANTS={'epde':'default','pdecomp':'pysindy_pdecomp_native',
          'deepmod':'deepmod_pdecomp_native','discover':'discover_pdecomp_native'}


def write_json(path,value):
    from support.discovery_lab.worker import _sanitize
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(_sanitize(value),sort_keys=True,allow_nan=False,separators=(',',':'))
    temp=path.with_suffix('.tmp');temp.write_text(text);temp.replace(path)


def prepare(method):
    from epde_bench.datasets import load
    from epde_bench.config import load_config
    from epde_bench import pdecomp,pdecomp_methods
    requests=[]
    for dataset,family in DATASETS.items():
        problem=load(dataset)
        base=(load_config(dataset,'default') if method=='epde' else
              pdecomp.describe(dataset,'native') if method=='pdecomp' else
              pdecomp_methods.describe(method,dataset,kind=problem.kind,
                                      orders=pdecomp.target_orders(problem)))
        for noise in (0.,2.):
            context=summarize_input(record_name=dataset,problem=problem,noise_percent=noise)
            context['data_features']=describe_problem(problem)
            choices=portfolio(method,base,context,hosted=True)
            if not choices or choices[0][0]!='rule':
                raise ValueError('Missing paired rule baseline for '+dataset+' / '+method)
            for seed in (0,1):
                for policy,label,settings in choices:
                    overrides=changes_for(method,base,settings)
                    if method!='epde':overrides={method:{'overrides':overrides}}
                    request={'method':method,'dataset':dataset,'family':family,'kind':problem.kind,
                             'noise':noise,'seed':seed,'policy':policy,'label':label,
                             'variant':VARIANTS[method],'overrides':overrides,
                             'settings':settings,'vector':vector(context,method,settings),
                             'semantic_signature':semantic_signature(method,base)}
                    request['input_id']=hashlib.sha256(json.dumps([dataset,noise,seed]).encode()).hexdigest()[:20]
                    request['run_id']=hashlib.sha256(json.dumps(request,sort_keys=True,default=str).encode()).hexdigest()[:24]
                    requests.append(request)
    return requests


def one(request_path,output):
    from epde_bench.runner import run_one
    request=json.loads(Path(request_path).read_text())
    if request['dataset'] not in DATASETS or request['method'] not in VARIANTS:
        raise ValueError('Only builtin research inputs are allowed')
    record=run_one(request['dataset'],request['variant'],request['noise'],request['seed'],
                   overrides=request['overrides'])
    record['execution_location']='github_actions'
    record['advisor_experiment']={k:request[k] for k in ('run_id','policy','label','family')}
    record['provenance']={'repository':'MakSoS1/EPDE','source_commit':os.environ.get('ADVISOR_SOURCE_SHA',''),
                          'workflow_run':os.environ.get('GITHUB_RUN_ID',''),
                          'input_policy':'allow-listed builtin synthetic data only'}
    write_json(output,record)


def execute(request,root,timeout):
    folder=root/request['run_id'];folder.mkdir(parents=True,exist_ok=True)
    config=folder/'request.json';result=folder/'result.json';log=folder/'run.log'
    write_json(config,request)
    if result.exists():return request,result
    start=time.monotonic()
    with log.open('w') as stream:
        proc=subprocess.Popen([sys.executable,__file__,'--one',str(config),'--output',str(result)],
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                               cwd=str(PIC.parents[1]))
        try:proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            write_json(result,{'status':'timeout','error':'Research per-run time limit reached',
                               'total_seconds':time.monotonic()-start})
        if not result.exists():
            write_json(result,{'status':'error','error':'Scientific child exited without a result',
                               'exit_code':proc.returncode,'total_seconds':time.monotonic()-start})
    return request,result


def run(method,out,workers,timeout):
    from support.discovery_lab.meta_advisor.training import quality_loss
    from epde_bench.identity import _source_hash
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    requests=prepare(method)
    provenance={'schema':'advisor-sweep/1','created_at':datetime.now(timezone.utc).isoformat(),
                'source_commit':os.environ.get('ADVISOR_SOURCE_SHA',''),
                'scientific_source_sha256':_source_hash(),'datasets':DATASETS,
                'noise_percent':[0,2],'seeds':[0,1],'method':method,'requested_runs':len(requests),
                'comparison':'same initial budget; Suggested v1 versus fixed portfolios',
                'timeout_seconds':timeout,'workers':workers}
    write_json(root/'campaign.json',provenance)
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(execute,r,root/'runs',timeout) for r in requests]
        for future in as_completed(pending):
            request,result=future.result();record=json.loads(result.read_text())
            row={k:request[k] for k in ('run_id','input_id','dataset','family','method','kind','noise',
                                       'seed','policy','label','vector','semantic_signature')}
            row.update(status=record.get('status','error'),loss=quality_loss(record),
                       seconds=float(record.get('fit_seconds') or record.get('total_seconds') or timeout),
                       effective_config=request['overrides'],
                       selected=record.get('selected'),metrics=record.get('metrics'),
                       error=record.get('error'),record_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
            rows.append(row)
            write_json(root/'rows.json',sorted(rows,key=lambda r:r['run_id']))
            print(json.dumps({'done':len(rows),'total':len(requests),'method':method,
                              'dataset':request['dataset'],'policy':request['policy'],
                              'status':row['status'],'loss':row['loss']},allow_nan=False),flush=True)
    write_json(root/'summary.json',{'runs':len(rows),'scored':sum(r['loss'] is not None for r in rows),
                                   'statuses':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=tuple(VARIANTS));parser.add_argument('--one')
    parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--timeout',type=int,default=75)
    args=parser.parse_args()
    if args.one:one(args.one,args.output)
    elif args.method:run(args.method,args.output,args.workers,args.timeout)
    else:parser.error('--method or --one is required')

if __name__=='__main__':main()
