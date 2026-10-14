"""Analyze frozen 756 EPDE runs; failed trials remain zero-accuracy."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import statistics

ARMS = ("native","anchor_utopia","anchor_min","mdl_utopia",
        "mdl_min","mdl_min_repair","no_mdl_min_repair")
RUNS = {0:"a2d0f22db1c87b8dd197bf024cc80aaa8ed6051c",
        1:"44e7d8d65bd67ae6ac5701a63b3a9e9dbc28c604",
        2:"9db33deb450a313708652404f1320c8993459bb8"}
COHORTS = {
"S1":["ac","burgers","duffing","lv","ode","vdp"],
"S2e":["nir1_decay","nir1_harmonic","nir1_logistic","nir1_quadratic",
       "nir1_cubic","nir1_advection","nir1_heat","nir1_reactiondiff",
       "nir1_wave","nir1_advdiff"],
"H11":["nir1_h11_gaussian_heat","nir1_h11_advection_decay",
       "nir1_h11_viscous_burgers","nir1_h11_klein_gordon",
       "nir1_h11_biharmonic","nir1_h11_airy","nir1_h11_shear",
       "nir1_h11_logistic_field","nir1_h11_gauss_reactdiff"],
"H14":["nir1_h14_damped_wave","nir1_h14_cole_hopf","nir1_h14_kpp_front",
       "nir1_h14_linear_ks","nir1_h14_beam","nir1_h14_airy_transport",
       "nir1_h14_cubic_saturation","nir1_h14_driven_advection",
       "nir1_h14_variable_diffusivity","nir1_h14_liouville_growth",
       "nir1_h14_hyperdiffusion_growth"]
}
COHORT = {d:c for c,ds in COHORTS.items() for d in ds}
SYSTEMS = sorted(COHORT)
assert len(SYSTEMS)==36
PAIRS = [
("PRIMARY MDL effect","mdl_min_repair","no_mdl_min_repair"),
("repair under MDL","mdl_min_repair","mdl_min"),
("MDL without repair","mdl_min","anchor_min"),
("MDL with utopia selector","mdl_utopia","anchor_utopia"),
("selector effect with MDL","mdl_min","mdl_utopia"),
("temporal prior vs native","anchor_utopia","native"),
]

def write_csv(path, rows):
    with open(path,"w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def exact(rec,metric):
    return int(rec.get("status")=="ok"
               and rec.get("metrics",{}).get(metric) is True)

def p_signflip(vals):
    # Exact system-clustered test for <=19 nonzero blocks; else seeded MC.
    vals=[v for v in vals if abs(v)>1e-12]
    if not vals: return 1.0,"exact",0
    from itertools import product
    obs=abs(sum(vals))
    n=len(vals)
    if n<=19:
        bigger=sum(abs(sum(s*v for s,v in zip(signs,vals)))>=obs-1e-12
                   for signs in product((-1,1),repeat=n))
        return bigger/(2**n),"exact",n
    rng=random.Random(1616);samples=100000; bigger=0
    for _ in range(samples):
        d=abs(sum(v if rng.randrange(2) else -v for v in vals))
        bigger+=d>=obs-1e-12
    return (bigger+1)/(samples+1),"MC-100000",n

def bootstrap(vals):
    rng=random.Random(1737);n=len(vals)
    data=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n
                for _ in range(8000))
    return data[200],data[7800]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    o=ap.parse_args()
    o.out.mkdir(parents=True,exist_ok=True)
    records={}
    conflicts=[]
    data_hashes=defaultdict(set)
    for f in o.input.rglob("native-full.json"):
        r=json.loads(f.read_text(encoding="utf8"))
        key=(r["dataset"],r["factorial_arm"],int(r["optimizer_seed"]))
        if key in records: raise RuntimeError("duplicate "+str(key))
        if key[0] not in SYSTEMS or key[1] not in ARMS or key[2] not in RUNS:
            raise RuntimeError("unknown identity "+str(key))
        if r.get("factorial_protocol")!="mdlablation20261014-v1":
            raise RuntimeError("wrong protocol "+str(key))
        if r.get("nir1_smoke_only") or r.get("data_seed_requested")!=0:
            raise RuntimeError("smoke/data seed "+str(key))
        sha=(f.parent/"source-sha.txt").read_text().strip()
        if sha!=RUNS[key[2]]: raise RuntimeError("source SHA differs "+str(key))
        if r.get("identity",{}).get("data_sha256") is None:
            raise RuntimeError("missing data SHA "+str(key))
        data_hashes[key[0]].add(r["identity"]["data_sha256"])
        evo=r.get("search_config",{}).get("evolution",{})
        expected_pop=32 if key[0]=="lv" else 16
        if evo.get("population_size")!=expected_pop or evo.get("training_epochs")!=5:
            conflicts.append("Wrong budget "+str(key)+" "+str(evo))
        records[key]=r
    expected={(d,a,s) for d in SYSTEMS for a in ARMS for s in RUNS}
    missing=expected-set(records)
    if missing: raise RuntimeError("MISSING "+str(len(missing))+" "+str(sorted(missing)[:9]))
    for d,h in data_hashes.items():
        if len(h)!=1: conflicts.append("Unequal inputs "+d+" "+str(h))
    rows=[]
    for (dataset,arm,seed),r in sorted(records.items()):
        rows.append(dict(dataset=dataset,cohort=COHORT[dataset],arm=arm,seed=seed,
            status=r.get("status"),exact=exact(r,"success_selected"),
            on_front=exact(r,"success_front"),
            hamming=r.get("metrics",{}).get("hamming_selected"),
            runtime_s=r.get("total_seconds"),
            error=str(r.get("error") or ""),
            data_sha=r["identity"]["data_sha256"],
            source_sha=RUNS[seed],
            identity_sha=r.get("identity",{}).get("sha256")))
    write_csv(o.out/"records_756.csv",rows)
    ag=[]
    for arm in ARMS:
        xs=[x for x in rows if x["arm"]==arm]
        ag.append(dict(arm=arm,records=len(xs),
            ok=sum(x["status"]=="ok" for x in xs),
            failed=sum(x["status"]!="ok" for x in xs),
            exact=sum(x["exact"] for x in xs),
            on_front=sum(x["on_front"] for x in xs)))
    write_csv(o.out/"aggregate_by_arm.csv",ag)
    detail=[]
    for dataset in SYSTEMS:
        row=dict(dataset=dataset,cohort=COHORT[dataset])
        for arm in ARMS:
            xs=[records[(dataset,arm,seed)] for seed in RUNS]
            row[arm+"_exact"]=sum(exact(x,"success_selected") for x in xs)
            row[arm+"_front"]=sum(exact(x,"success_front") for x in xs)
            row[arm+"_failure"]=sum(x.get("status")!="ok" for x in xs)
        detail.append(row)
    write_csv(o.out/"per_system_36.csv",detail)
    comparisons=[]
    for name,a,b in PAIRS:
        differences=[]
        win=loss=tie=0
        for d in SYSTEMS:
            vals=[]
            for seed in RUNS:
                aa=exact(records[(d,a,seed)],"success_selected")
                bb=exact(records[(d,b,seed)],"success_selected")
                vals.append(aa-bb)
                win+=aa>bb
                loss+=aa<bb
                tie+=aa==bb
            differences.append(sum(vals)/3)
        p,mode,nonzero=p_signflip(differences)
        lo,hi=bootstrap(differences)
        comparisons.append(dict(name=name,A=a,B=b,
            delta_pp=100*statistics.mean(differences),
            wins=win,losses=loss,ties=tie,
            p_system_signflip=p,p_mode=mode,
            n_nonzero_systems=nonzero,
            ci95_low_pp=100*lo,ci95_high_pp=100*hi))
    write_csv(o.out/"paired_comparisons.csv",comparisons)
    failures=Counter((x["arm"],x["error"]) for x in rows if x["status"]!="ok")
    byfail=defaultdict(list)
    for x in rows:
        if x["status"]!="ok": byfail[(x["arm"],x["error"])].append(
            f"{x['dataset']}/s{x['seed']}")
    failtext=["# Failure audit","",
      "All failures count as structural failures, not removed from denominators.","",
      "| Arm | Number | Error |","|---|---:|---|"]
    for (arm,err),count in sorted(failures.items()):
        failtext.append(f"| {arm} | {count} | {err.replace('|','/')[:180]} |")
    failtext.append("")
    for (arm,err),keys in sorted(byfail.items()):
        failtext.append(f"### {arm}: {err}")
        failtext.append(", ".join(keys))
        failtext.append("")
    (o.out/"FAILURES_RU.md").write_text("\n".join(failtext)+"\n",encoding="utf8")
    report=[
      "# MDL causal ablation: actual 756 EPDE full searches",
      "",
      "All 36 physics problems were previously seen during H9-H14 development.",
      "This is a repeated-seed causal ablation, NOT a fresh physical holdout.",
      "",
      f"Total: {len(rows)}/756 identities; errors: {sum(x['status']!='ok' for x in rows)}.",
      f"Identity/budget warnings: {len(conflicts)}; full details in summary.json.",
      "",
      "## Exact structural recovery (all 3 seeds, crashes count as failure)","",
      "| Arm | Exact /108 | Truth on Pareto /108 | Crashes /108 |",
      "|---|---:|---:|---:|"]
    for a in ag:
        report.append(f"| {a['arm']} | {a['exact']}/108 ({a['exact']/108:.1%}) | {a['on_front']}/108 | {a['failed']}/108 |")
    report += ["","## Paired causal contrasts","",
               "Two-sided sign-flip clustered over 36 physical systems; three optimizer seeds treated as correlated replicates. Cluster bootstrap 95% CI.",
               "",
               "| Comparison A-B | Delta pp | Wins / Losses / Ties (108 pairs) | p | cluster CI 95%, pp |",
               "|---|---:|---|---:|---|"]
    for c in comparisons:
        report.append(f"| {c['name']} | {c['delta_pp']:+.2f} | {c['wins']}/{c['losses']}/{c['ties']} | {c['p_system_signflip']:.5f} ({c['p_mode']}) | [{c['ci95_low_pp']:+.2f}, {c['ci95_high_pp']:+.2f}] |")
    report += ["","## Cohorts (NOT independent heldouts)","",
               "| Cohort | n laws | native | MDL+repair | noMDL+repair |",
               "|---|---:|---:|---:|---:|"]
    for label,names in COHORTS.items():
        totals=[sum(exact(records[(d,a,s)],"success_selected")
                    for d in names for s in RUNS)
                for a in ("native","mdl_min_repair","no_mdl_min_repair")]
        report.append(f"| {label} | {len(names)} | {totals[0]}/{3*len(names)} | {totals[1]}/{3*len(names)} | {totals[2]}/{3*len(names)} |")
    report += ["","## Every physical system: exact counts out of 3","",
               "| Dataset | Cohort | native | H9 min | H11 | H12 MDL | H12 no MDL |",
               "|---|---|---:|---:|---:|---:|"]
    for r in detail:
        report.append(f"| {r['dataset']} | {r['cohort']} | {r['native_exact']} | {r['anchor_min_exact']} | {r['mdl_min_exact']} | {r['mdl_min_repair_exact']} | {r['no_mdl_min_repair_exact']} |")
    report += ["","## Interpretation limits","",
      "- Primary contrast changes ONLY whether MDL fitness is enabled, with H9, H11, H12 otherwise fixed.",
      "- Original H9/H11 failures are recorded and not silently repaired or retried. Repair arms independently use the already implemented H12 recovery.",
      "- Default means PIC benchmark fork baseline, not clean ITMO upstream.",
      "- All evaluated law types were previously used during development; optimizer seeds 1 and 2 are new but new random seeds do NOT turn systems into external holdout.",
      "- One data seed, zero noise, synthetic trajectories, no true out-of-trajectory prediction validation.",
      "- Exact structure ignores coefficient accuracy; supplementary coefficient and physical residual checks required.",
      "- Source code differs across the three runs only by added workflow files, not search logic.",
      "","Files: records_756.csv, per_system_36.csv, paired_comparisons.csv, FAILURES_RU.md, summary.json."]
    (o.out/"REPORT_RU.md").write_text("\n".join(report)+"\n",encoding="utf8")
    summary=dict(total=len(rows),errors=sum(x["status"]!="ok" for x in rows),
                 warnings=conflicts,by_arm=ag,comparisons=comparisons,
                 error_groups=[dict(arm=k[0],error=k[1],n=n) for k,n in sorted(failures.items())])
    (o.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if conflicts: raise RuntimeError("Identity/budget violations found")

if __name__=="__main__": 
    main()
