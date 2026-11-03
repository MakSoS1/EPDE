"""Read-only, checksummed examples from allow-listed synthetic experiments.

No visitor directory is scanned and no public result is copied into a private
queue. The publication script is the only writer of this versioned library.
"""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from .portable import input_key

PUBLIC_DIR=Path(__file__).resolve().parents[3]/'public_results'
METHODS={'epde','pdecomp','deepmod','discover'}


def load_library(directory=PUBLIC_DIR):
    root=Path(directory).resolve()
    index=root/'index.json'
    if not index.is_file():
        return []
    if index.stat().st_size>100_000:
        raise ValueError('Public example index exceeds the size limit')
    manifest=json.loads(index.read_text(encoding='utf-8'))
    groups=manifest.get('groups')
    if manifest.get('schema')!='epde-public-results/1' or not isinstance(groups,list) or len(groups)>20:
        raise ValueError('Invalid public example index')
    from epde_bench.datasets import REGISTRY
    result=[];identifiers=set()
    for group in groups:
        group=dict(group);slug=group.get('id','');runs=[];keys=set()
        if not re.fullmatch(r'[a-z0-9_-]{1,80}',slug) or slug in identifiers:
            raise ValueError('Invalid or duplicate public example identifier')
        identifiers.add(slug)
        entries=group.get('runs')
        if not isinstance(entries,list) or not 1<=len(entries)<=12:
            raise ValueError('Invalid public example run list')
        seen=set()
        for entry in entries:
            name=entry.get('file','')
            if not re.fullmatch(r'[a-z0-9_-]{1,100}\.json',name) or name in seen:
                raise ValueError('Invalid public example file path')
            seen.add(name);path=root/name
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError('Public example file is unavailable')
            if path.stat().st_size>1_000_000:
                raise ValueError('Public example exceeds the size limit')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=entry.get('sha256'):
                raise ValueError('Public example checksum mismatch')
            record=json.loads(raw)
            if (record.get('dataset') not in REGISTRY or record.get('method') not in METHODS
                    or (record.get('problem') or {}).get('source')!='synthetic'
                    or 'data_file' in record or not (record.get('provenance') or {}).get('workflow_run')):
                raise ValueError('Only provenance-bearing builtin synthetic runs may be public')
            key=input_key(record)
            if not key:raise ValueError('Public example lacks a scientific input fingerprint')
            keys.add(key)
            runs.append(dict(entry,id='public-'+slug+'-'+name[:-5],record=record,raw=raw))
        if len(keys)!=1:
            raise ValueError('Public group mixes different numerical inputs')
        group.update(runs=runs,input_key=next(iter(keys)))
        result.append(group)
    return result


def jobs_and_results(groups):
    """Adapt public records to the existing result renderer, without writes."""
    jobs=[];records={}
    for group in groups:
        for entry in group['runs']:
            record=entry['record'];identifier=entry['id']
            try:created=datetime.fromisoformat(record.get('started','').replace('Z','+00:00')).timestamp()
            except (ValueError,TypeError):created=0.
            seconds=float(record.get('fit_seconds') or record.get('total_seconds') or 0.)
            jobs.append({'id':identifier,'method':record['method'],'dataset':record['dataset'],
                         'variant':record.get('variant'),'noise':record['noise'],'seed':record['seed'],
                         'label':(record.get('problem') or {}).get('title',record['dataset']),
                         'state':'finished' if record.get('status')=='ok' else record.get('status','error'),
                         'created':created,'started':created,'ended':created+seconds,
                         'public_label':entry.get('label',record['method']),
                         'public_group':group['id'],'public':True})
            records[identifier]=record
    return jobs,records
