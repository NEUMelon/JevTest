"""Rebuild auditable EM results from retained responses, without network access.

Run from jevwrangle: python -m src.recovery.offline --output runs/recovery_20260930
Existing output directories are rejected. Legacy artifacts are never overwritten.
"""
import argparse
import hashlib
import json
import platform
import sqlite3
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.special import expit, softmax
from scipy.optimize import minimize
from .core import noul, score, choice, distribution, llm_probability, logit, metrics, fit_aggregator
from src.experiments.e2_attribution import compute_code_features, POLARITIES
from src.experiments.magellan_feats import compute_magellan_features
from src.experiments.e8_error_taxonomy import classify_pair_taxonomy

BASE = Path(__file__).resolve().parents[2]
DATASETS = ["wa", "ag", "da", "ab"]


class Recovery:
    def __init__(self, output):
        self.out = output
        self.out.mkdir(parents=True, exist_ok=False)
        for folder in ["features", "predictions", "models", "tables", "oof", "source_snapshot"]:
            (self.out / folder).mkdir()
        for rel in ['src/recovery/core.py','src/recovery/offline.py','src/experiments/e2_attribution.py',
                    'src/experiments/magellan_feats.py','src/experiments/e8_error_taxonomy.py']:
            target=self.out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((BASE/rel).read_bytes())
        self.db = sqlite3.connect((BASE / "cache/calls.sqlite").as_uri()+"?mode=ro", uri=True)
        self.inputs, self.issues, self.missing, self.metric_rows, self.slice_rows = {}, [], [], [], []
        self.calibration_rows, self.bin_rows = [], []
        # New observations are separate from the historical cache and are only
        # eligible for the explicitly identified missing/conflicting requests.
        repairs=BASE/'runs/api_jev_repairs_20260930/observations.json'
        self.repairs=json.loads(repairs.read_text(encoding='utf8')) if repairs.exists() else []

    def track(self, path):
        self.inputs[path.relative_to(BASE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def pairs(self, path):
        rows = [json.loads(s) for s in self.track(path).read_text(encoding="utf8").splitlines()]
        ids = [p["pair_id"] for p in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate pair IDs: {path}")
        return rows

    def csv(self, path):
        return pd.read_csv(self.track(path))

    def questions(self, ds, name):
        return yaml.safe_load(self.track(BASE / f"questions/{ds}/{name}.yaml").read_text(encoding="utf8"))

    def cached(self, payload):
        key = hashlib.sha256(json.dumps(["aihubmix", payload], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        row = self.db.execute("SELECT response, created_at FROM calls WHERE key=?", (key,)).fetchone()
        return (json.loads(row[0]), key, row[1]) if row else None

    def jev(self, ds, split, pairs, questions, legacy, decomposed=False):
        result = []
        for i, p in enumerate(pairs):
            row = dict(pair_id=p["pair_id"], label=p["label"])
            candidates = []
            for alias in ["jev-latest", "jev-1.13"]:
                payload = dict(model=alias, state={"record_a":p["record_a"], "record_b":p["record_b"]}, questions=questions)
                hit = self.cached(payload)
                if hit:
                    resp, key, timestamp = hit
                    try:
                        a = resp["answers"]
                        decoded = {"jev_"+q:noul(a[q]) for q in questions} if decomposed else {
                            "p_h1n":noul(a["H1_noul"]), "p_h1s":score(a["H1_score"]),
                            "p_h1c":choice(a["H1_choice"]), "p_h0":noul(a["H0"]),
                            **{"score_"+k:float(v) for k,v in a["H1_score"]["probabilities"].items()},
                            **{"choice_"+k:float(v) for k,v in a["H1_choice"]["probabilities"].items()}}
                        comparison = list(decoded) if decomposed else ["p_h1n"]
                        matched = all(abs(decoded[c]-float(legacy.iloc[i][c]))<1e-9 for c in comparison)
                        candidates.append((matched, decoded, alias, key, timestamp, resp.get("model")))
                    except (ValueError, KeyError, TypeError) as e:
                        self.issues.append(dict(dataset=ds, split=split, pair_id=p["pair_id"], kind="response_decode", detail=str(e)))
            chosen = next((r for r in candidates if r[0]), None)
            repaired=next((r for r in self.repairs if r['dataset']==ds and r['split']==split and
                r['pair_id']==p['pair_id'] and (('same_brand' in r['response'].get('answers',{}) or 'same_title' in r['response'].get('answers',{}) or 'same_manufacturer' in r['response'].get('answers',{}))==decomposed)),None)
            if chosen is None and repaired:
                resp=repaired['response'];a=resp['answers'];meta=repaired['meta']
                if resp.get('model')!='typesafe/jev-1.13-20260917':raise ValueError('Repair version drift')
                decoded={"jev_"+q:noul(a[q]) for q in questions} if decomposed else {
                    "p_h1n":noul(a["H1_noul"]),"p_h1s":score(a["H1_score"]),"p_h1c":choice(a["H1_choice"]),"p_h0":noul(a["H0"]),
                    **{"score_"+k:float(v) for k,v in a["H1_score"]["probabilities"].items()},
                    **{"choice_"+k:float(v) for k,v in a["H1_choice"]["probabilities"].items()}}
                row.update(decoded,status='new_repair_observation',requested_model='jev-1.13',response_model=resp['model'],
                           cache_key=meta['key'],request_id=meta['request_id'])
                self.issues.append(dict(dataset=ds,split=split,pair_id=p['pair_id'],kind='new_observation_replaces_missing_legacy_evidence',
                                        feature_group='decomposed' if decomposed else 'holistic'))
                result.append(row);continue
            if chosen:
                _, decoded, alias, key, timestamp, model = chosen
                row.update(decoded, status="verified_cache", requested_model=alias, response_model=model,
                           cache_key=key, cache_created_at=timestamp)
            else:
                kind = "legacy_cache_conflict" if candidates else "missing_cached_response"
                row["status"] = kind
                self.issues.append(dict(dataset=ds, split=split, pair_id=p["pair_id"], kind=kind,
                                        feature_group="decomposed" if decomposed else "holistic"))
                self.missing.append(dict(dataset=ds, split=split, pair_id=p["pair_id"], reason=kind,
                                         payload=dict(model="jev-1.13", state={"record_a":p["record_a"],"record_b":p["record_b"]}, questions=questions)))
            result.append(row)
        df = pd.DataFrame(result)
        df.to_csv(self.out / f"features/{ds}_{split}_{'D' if decomposed else 'H'}.csv", index=False)
        return df

    def llm(self, ds, pairs, old):
        rows=[]
        for i,p in enumerate(pairs):
            prompt = f"Record A: {json.dumps(p['record_a'])}\nRecord B: {json.dumps(p['record_b'])}\nDo Record A and Record B refer to the same real-world item? Output strictly in JSON format: {{\"answer\": \"yes\" or \"no\", \"match_probability\": <probability from 0.0 to 1.0>}}"
            row=dict(pair_id=p["pair_id"], label=p["label"])
            for alias, col, budgets in [("gpt-6-luna","p_luna",[700,600,1200]),("deepseek-v4-flash","p_deepseek",[300,600,1200])]:
                hits=[]
                for tokens in budgets:
                    payload=dict(model=alias,messages=[{"role":"user","content":prompt}],temperature=0.0,
                                 max_tokens=tokens,response_format={"type":"json_object"})
                    if "luna" in alias: payload["reasoning_effort"]="low"
                    hit=self.cached(payload)
                    if hit:
                        resp,key,ts=hit
                        try:
                            c=resp["choices"][0]
                            if c.get("finish_reason")=="length": continue
                            prob=llm_probability(c["message"]["content"])
                            hits.append((prob,key,resp.get("model"),ts))
                        except (ValueError,KeyError,TypeError): pass
                # Matching identifies which retained response underlies the old run;
                # an unmatched raw probability may have been changed by legacy parsing.
                h=next((v for v in hits if abs(v[0]-float(old.iloc[i][col]))<1e-9),hits[0] if hits else None)
                if h:
                    row.update({col:h[0],col+"_cache_key":h[1],col+"_model":h[2],col+"_status":"verified_cache"})
                    if abs(h[0]-float(old.iloc[i][col]))>1e-9:
                        self.issues.append(dict(dataset=ds,pair_id=p["pair_id"],kind="legacy_llm_probability_changed", model=alias))
                else:
                    row[col+"_status"]="missing_reported_probability_or_response"
                    self.issues.append(dict(dataset=ds,pair_id=p["pair_id"],kind=row[col+"_status"],model=alias))
            rows.append(row)
        df=pd.DataFrame(rows)
        df.to_csv(self.out/f"features/{ds}_test_LLM.csv",index=False)
        return df

    def evaluate(self, ds, system, budget, seed, records, p, threshold, note="", parameters=None):
        y=np.array([r["label"] for r in records]);p=np.asarray(p,dtype=float)
        finite=np.isfinite(p); n=int(finite.sum()); complete=bool(finite.all())
        status="recomputed_complete" if complete else "diagnostic_partial_BLOCKED"
        pred=np.where(finite,(p>=threshold).astype(int),-1)
        df=pd.DataFrame(dict(pair_id=[r["pair_id"] for r in records],label=y,probability=p,
                             prediction=pred,threshold=threshold,status=np.where(finite,"available","missing")))
        slug=f"{ds}_{system}_b{budget}_s{seed}"
        df.to_csv(self.out/f"predictions/{slug}.csv",index=False)
        m=metrics(y[finite],p[finite],threshold) if n else {}
        self.metric_rows.append(dict(dataset=ds,system=system,budget=budget,seed=seed,n_expected=len(y),
                                     n_available=n,coverage=n/len(y),status=status,threshold=threshold,
                                     f1_at_05=metrics(y[finite],p[finite])["f1"] if n else np.nan,
                                     **m,**(parameters or {}),note=note))
        if budget in [0,200]:
            categories=np.array([classify_pair_taxonomy(ds,r["record_a"],r["record_b"],r["label"]) for r in records])
            counts=[]
            for cat in sorted(set(categories)):
                mask=(categories==cat)&finite
                if mask.any():
                    sm=metrics(y[mask],p[mask],threshold)
                    counts.append(sm)
                    self.slice_rows.append(dict(dataset=ds,system=system,budget=budget,seed=seed,slice=cat,
                                                 n_expected=int(np.sum(categories==cat)),n_available=int(mask.sum()),
                                                 status=status,**sm,note="Heuristic slices; label-noise name is not a verified annotation"))
            for key in ["tp","fp","fn","tn"]:
                assert sum(s[key] for s in counts)==m.get(key,0), "Slice totals must reproduce the main prediction file"
        return slug

    def calibration(self, ds, pool, test):
        for system, cols, match in [("J-H1n",["p_h1n"],0), ("J-H1s",["score_0","score_1","score_2"],2),
                                    ("J-H1c",["choice_same","choice_different","choice_cannot_tell"],0)]:
            valid_train=np.isfinite(pool[cols]).all(axis=1);valid_test=np.isfinite(test[cols]).all(axis=1)
            if not valid_train.all() or not valid_test.all():
                self.issues.append(dict(dataset=ds,kind="calibration_incomplete_BLOCKED",system=system));continue
            tr=pool[cols].values;te=test[cols].values;y=pool.label.values;yt=test.label.values
            def transform(a,T):
                if len(cols)==1:return expit(logit(a[:,0])/T)
                return softmax(np.log(np.clip(a,1e-4,1))/T,axis=1)[:,match]
            def nll(y,p):
                p=np.clip(p,1e-4,1-1e-4)
                return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
            fit=minimize(lambda t:nll(y,transform(tr,float(t[0]))),x0=[1.],method="L-BFGS-B",bounds=[(.05,20.)])
            if not fit.success:raise RuntimeError(f"Temperature fit failed: {ds} {system} {fit.message}")
            T=float(fit.x[0])
            pd.DataFrame(dict(pair_id=test.pair_id,label=yt,raw=te[:,match],scaled=transform(te,T))).to_csv(self.out/f"predictions/{ds}_{system}_calibration.csv",index=False)
            for mode,p in [("raw",te[:,match]),("temperature",transform(te,T))]:
                eces={}
                for binning,bins in [("equal_mass",15),("equal_width",10)]:
                    if binning=="equal_mass":groups=np.array_split(np.argsort(p,kind="stable"),bins)
                    else:
                        index=np.minimum((p*bins).astype(int),bins-1)
                        groups=[np.flatnonzero(index==b) for b in range(bins)]
                    ece=0
                    for b,g in enumerate(groups):
                        avgp=float(p[g].mean()) if len(g) else np.nan
                        avgy=float(yt[g].mean()) if len(g) else np.nan
                        if len(g):ece+=len(g)/len(p)*abs(avgp-avgy)
                        self.bin_rows.append(dict(dataset=ds,system=system,mode=mode,binning=binning,bin=b,
                                                  count=len(g),mean_probability=avgp,positive_fraction=avgy))
                    eces["ece_"+binning]=float(ece)
                self.calibration_rows.append(dict(dataset=ds,system=system,mode=mode,T=T if mode=="temperature" else 1,
                    **eces,brier=float(np.mean((p-yt)**2)),nll=nll(yt,p),n_test=len(p),n_fit=len(y),
                    status="recomputed_complete",note="T fits binary match NLL on pool labels; full softmax for Score/Choice; floor=1e-4; no T confidence interval yet"))

    def run(self, budgets):
        for ds in DATASETS:
            print(f"RECOVER {ds}: cache provenance",flush=True)
            records={"test":self.pairs(BASE/f"data/canonical/{ds}/test.jsonl"),
                     "pool":self.pairs(BASE/f"data/pools/{ds}_pool2000.jsonl")}
            if "full" in budgets:records["full"]=self.pairs(BASE/f"data/canonical/{ds}/train.jsonl")
            oldtest=self.csv(BASE/f"runs/e1_preds_{ds}.csv").set_index("pair_id",verify_integrity=True).loc[[p["pair_id"] for p in records["test"]]].reset_index()
            if oldtest.label.tolist()!=[p["label"] for p in records["test"]]:raise ValueError("Legacy labels disagree with canonical records")
            H={};D={};C={};M={}
            for split,prs in records.items():
                oldh=oldtest if split=="test" else self.csv(BASE/f"runs/e2_jev_h_{ds}_{split}.csv")
                H[split]=self.jev(ds,split,prs,self.questions(ds,"holistic"),oldh)
                D[split]=self.jev(ds,split,prs,self.questions(ds,"decomposed"),self.csv(BASE/f"runs/e2_jev_d_{ds}_{split}.csv"),True)
                C[split]=compute_code_features(ds,prs).values
                M[split]=compute_magellan_features(ds,prs).values
                pd.DataFrame(M[split]).assign(pair_id=[p["pair_id"] for p in prs]).to_csv(self.out/f"features/{ds}_{split}_M.csv",index=False)
            L=self.llm(ds,records["test"],oldtest)
            for sysname,col in [("J-H0","p_h0"),("J-H1n","p_h1n"),("J-H1s","p_h1s"),("J-H1c","p_h1c")]:
                self.evaluate(ds,sysname,0,0,records["test"],H["test"][col].values,.5,
                              note="H0 retains the legacy state/serialization deviation")
            for sysname,col in [("L1-H1","p_luna"),("L2-H1","p_deepseek")]:
                self.evaluate(ds,sysname,0,0,records["test"],L[col].values,.5,note="Reported verbal probability; no invented fallback")
            dcols=["jev_"+q for q in POLARITIES[ds]]
            def logits_missing(a):
                out=np.full_like(a,np.nan,dtype=float);finite=np.isfinite(a)
                out[finite]=logit(a[finite]);return out
            X={}
            for split in records:
                h=logits_missing(H[split][["p_h1n"]].values)
                d=logits_missing(D[split].reindex(columns=dcols).values)
                X[split]={"J-H1+LR":h,"J-D+LR":d,"J-D+C+LR":np.column_stack([d,C[split]]),
                          "C+LR":C[split],"M+LR":M[split]}
            d0=expit(np.mean(X["test"]["J-D+LR"]*np.array(list(POLARITIES[ds].values())),axis=1))
            self.evaluate(ds,"J-D0",0,0,records["test"],d0,.5)
            self.calibration(ds,H["pool"],H["test"])
            for budget in budgets:
                seeds=3 if budget=="full" else (5 if int(budget)==1000 else 10)
                for seed in range(seeds):
                    split="full" if budget=="full" else "pool"
                    prs=records[split]; ids=[p["pair_id"] for p in prs]
                    if budget=="full":idx=np.arange(len(prs))
                    else:
                        path=self.track(BASE/f"data/budgets/{ds}/b{budget}_seed{seed}.json")
                        wanted=json.loads(path.read_text())["pair_ids"]
                        if len(wanted)!=int(budget) or len(set(wanted))!=len(wanted) or not set(wanted)<=set(ids):
                            raise ValueError("Frozen budget IDs missing, repeated, or wrong-sized")
                        idx=np.flatnonzero(np.isin(ids,wanted))
                    y=np.array([p["label"] for p in prs])[idx]
                    for system in X[split]:
                        train=X[split][system][idx];test=X["test"][system]
                        if not np.isfinite(train).all():
                            self.issues.append(dict(dataset=ds,system=system,budget=budget,seed=seed,kind="training_missing_feature_BLOCKED"));continue
                        model,thr,oof,params=fit_aggregator(train,y,len(idx),seed)
                        p=np.full(len(test),np.nan);finite=np.isfinite(test).all(axis=1)
                        p[finite]=model.predict_proba(test[finite])[:,1]
                        slug=self.evaluate(ds,system,budget,seed,records["test"],p,thr,
                            note="Magellan-style feature fallback; M+best matcher selection remains pending" if system=="M+LR" else "",parameters={"C":params["C"]})
                        joblib.dump(model,self.out/f"models/{slug}.joblib")
                        pd.DataFrame(dict(pair_id=np.array(ids)[idx],label=y,oof_probability=oof)).to_csv(self.out/f"oof/{slug}.csv",index=False)
                        (self.out/f"models/{slug}.json").write_text(json.dumps(dict(threshold=thr,**params),indent=2),encoding="utf8")
                print(f"RECOVER {ds}: b={budget} completed",flush=True)
            self.save()
        self.db.close()
        self.track(BASE/"cache/calls.sqlite")
        if self.repairs:
            self.track(BASE/'runs/api_jev_repairs_20260930/observations.json')
        for path in (BASE/"src/recovery").glob("*.py"):self.track(path)
        self.track(BASE/"src/experiments/e2_attribution.py")
        self.track(BASE/"src/experiments/magellan_feats.py")
        self.track(BASE/"src/experiments/e8_error_taxonomy.py")
        config=dict(status="RECOVERED_NOT_FROZEN",python=sys.executable,python_version=platform.python_version(),
                    numpy=np.__version__,sklearn=sklearn.__version__,pandas=pd.__version__,budgets=budgets,
                    api_calls=0,run_type="offline recovery from retained cache; historical evidence limitations retained",
                    blocked_experiments=["Adult supervised E7", "E5 P2/P5", "E6 fair latency/cost", "LLM LR prompt consistency", "LLM decomposition", "Qwen", "Ditto", "decision-model-preview", "M+best"],
                    inputs=self.inputs)
        (self.out/"manifest.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf8")
        (self.out/"repair_requests.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in self.missing),encoding="utf8")
        print(f"RECOVERY_DONE {self.out} issues={len(self.issues)}",flush=True)

    def save(self):
        df=pd.DataFrame(self.metric_rows)
        df.to_csv(self.out/"tables/metrics_by_seed.csv",index=False)
        keys=["dataset","system","budget","status"]
        df.groupby(keys,dropna=False).agg(f1_mean=("f1","mean"),f1_std=("f1","std"),
            n_seeds=("seed","count"),coverage=("coverage","min"),f1_at_05_mean=("f1_at_05","mean")).reset_index().to_csv(self.out/"tables/em_summary.csv",index=False)
        pd.DataFrame(self.slice_rows).to_csv(self.out/"tables/error_slices_by_seed.csv",index=False)
        pd.DataFrame(self.calibration_rows).to_csv(self.out/"tables/calibration.csv",index=False)
        pd.DataFrame(self.bin_rows).to_csv(self.out/"tables/calibration_bins.csv",index=False)
        (self.out/"issues.json").write_text(json.dumps(self.issues,ensure_ascii=False,indent=2),encoding="utf8")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--budgets",nargs="+",default=["50","200","1000","full"])
    args=ap.parse_args()
    if any(b not in ["50","200","1000","full"] for b in args.budgets):ap.error("Unsupported budget")
    Recovery(args.output.resolve()).run(args.budgets)


if __name__=="__main__":main()
