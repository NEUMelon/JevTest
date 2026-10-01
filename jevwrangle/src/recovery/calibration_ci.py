"""Bootstrap intervals for repaired calibration; fit temperatures on pool only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, softmax
from threadpoolctl import threadpool_limits
from .core import logit

BASE = Path(__file__).resolve().parents[2]
RUN = BASE / 'runs/recovery_repaired_20260930'
OUT = BASE / 'runs/calibration_ci_20260930'


def metric_batch(y, p):
    """Each row is a bootstrap sample, including its own bin boundaries."""
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim == 1: y, p = y[None, :], p[None, :]
    n = p.shape[1]; clipped = np.clip(p, 1e-4, 1-1e-4)
    result = dict(brier=np.mean((p-y)**2, axis=1),
                  nll=-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped), axis=1))
    order = np.argsort(p, axis=1, kind='stable')
    sorted_p = np.take_along_axis(p, order, axis=1); sorted_y = np.take_along_axis(y, order, axis=1)
    ece = np.zeros(len(p))
    for g in np.array_split(np.arange(n), 15):
        if len(g): ece += len(g)/n*np.abs(sorted_p[:, g].mean(axis=1)-sorted_y[:, g].mean(axis=1))
    result['ece_equal_mass'] = ece
    bins = np.minimum((p*10).astype(int), 9); ece = np.zeros(len(p))
    for b in range(10):
        mask = bins == b
        # Weighted count*absolute mean difference equals absolute residual sum.
        ece += np.abs(np.sum((p-y)*mask, axis=1))/n
    result['ece_equal_width'] = ece
    return result


def transformed(values, match, T):
    return expit(logit(values[:, 0])/T) if values.shape[1] == 1 else softmax(np.log(np.clip(values, 1e-4, 1))/T, axis=1)[:, match]


def temperature_fit(values, y, match, weights=None):
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    def objective(t):
        p = np.clip(transformed(values, match, float(t[0])), 1e-4, 1-1e-4)
        return float(np.sum(weights*(-y*np.log(p)-(1-y)*np.log(1-p)))/weights.sum())
    f = minimize(objective, [1.], method='L-BFGS-B', bounds=[(.05, 20.)])
    return float(f.x[0]), bool(f.success), str(f.message)


@threadpool_limits.wrap(limits=1)
def main():
    OUT.mkdir(exist_ok=False); rows, temperature_rows, input_hashes = [], [], {}
    (OUT/'source_snapshot').mkdir()
    for name in ['calibration_ci.py', 'core.py']:
        p = BASE/'src/recovery'/name; (OUT/'source_snapshot'/name).write_bytes(p.read_bytes())
    point = pd.read_csv(RUN/'tables/calibration.csv')
    input_hashes['tables/calibration.csv'] = hashlib.sha256((RUN/'tables/calibration.csv').read_bytes()).hexdigest()
    for ds in ['wa', 'ag', 'da', 'ab']:
        pool = pd.read_csv(RUN/f'features/{ds}_pool_H.csv')
        test = pd.read_csv(RUN/f'features/{ds}_test_H.csv')
        for split in ['pool', 'test']:
            p = RUN/f'features/{ds}_{split}_H.csv'; input_hashes[p.relative_to(RUN).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        for system, cols, match in [('J-H1n', ['p_h1n'], 0), ('J-H1s', ['score_0','score_1','score_2'], 2),
                                   ('J-H1c', ['choice_same','choice_different','choice_cannot_tell'], 0)]:
            raw = pool[cols].values; y = pool.label.values
            if not np.isfinite(raw).all(): raise ValueError('Incomplete calibration pool')
            # Quantized response vectors plus labels are sufficient statistics
            # for this NLL. Multinomial resampling is paired-row bootstrap.
            cases, counts = np.unique(np.column_stack([raw, y]), axis=0, return_counts=True)
            T, success, message = temperature_fit(cases[:, :-1], cases[:, -1], match, counts)
            if not success: raise RuntimeError(message)
            expected = point[(point.dataset==ds)&(point.system==system)&(point['mode']=='temperature')].iloc[0]
            if not np.isclose(T, expected['T'], atol=.002): raise ValueError('Temperature point estimate failed reproduction')
            rng = np.random.default_rng(20260930)
            temperatures, failures = [], []
            for replicate in range(2000):
                weights = rng.multinomial(len(pool), counts/counts.sum())
                fitted, ok, reason = temperature_fit(cases[:, :-1], cases[:, -1], match, weights)
                if ok: temperatures.append(fitted)
                else: failures.append(dict(replicate=replicate, T=fitted, reason=reason))
            np.save(OUT/f'{ds}_{system}_T_bootstrap.npy', np.asarray(temperatures))
            (OUT/f'{ds}_{system}_T_optimizer_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf8')
            low, high = np.percentile(temperatures, [2.5,97.5]) if temperatures else [np.nan,np.nan]
            temperature_rows.append(dict(dataset=ds, system=system, T=float(expected['T']), T_ci_low=low, T_ci_high=high,
                n_bootstrap_success=len(temperatures), n_bootstrap_failed=len(failures),
                at_lower_bound_fraction=float(np.mean(np.isclose(temperatures,.05))),
                at_upper_bound_fraction=float(np.mean(np.isclose(temperatures,20))),
                status='COMPLETE_NOT_FROZEN' if not failures else 'OPTIMIZER_FAILURES_REVIEW_REQUIRED'))
            yt = test.label.values
            for mode, p in [('raw', test[cols].values[:,match]), ('temperature', transformed(test[cols].values,match,float(expected['T'])))]:
                measured = metric_batch(yt,p)
                expected_metric = point[(point.dataset==ds)&(point.system==system)&(point['mode']==mode)].iloc[0]
                for key, value in measured.items():
                    if not np.isclose(value[0],expected_metric[key],atol=1e-9):raise ValueError(f'Metric reproduction failure {key}')
                rng = np.random.default_rng(20260930); samples = {key:[] for key in measured}
                for start in range(0,2000,100):
                    idx = rng.integers(0,len(p),size=(100,len(p)))
                    result = metric_batch(yt[idx],p[idx])
                    for key,value in result.items():samples[key].extend(value)
                for key,sample in samples.items():
                    lo,hi=np.percentile(sample,[2.5,97.5])
                    rows.append(dict(dataset=ds,system=system,mode=mode,metric=key,estimate=measured[key][0],ci_low=lo,ci_high=hi,
                        replicates=2000,n_test=len(p),status='COMPLETE_CONDITIONAL_ON_FITTED_T',
                        note='Test-row bootstrap holds training-fitted T fixed; pool bootstrap T interval reported separately; equal-mass ties stable'))
            pd.DataFrame(rows).to_csv(OUT/'metric_intervals.csv',index=False)
            pd.DataFrame(temperature_rows).to_csv(OUT/'temperature_intervals.csv',index=False)
            print(f'CALIBRATION_CI {ds} {system} T={T:.4f} [{low:.4f},{high:.4f}] failures={len(failures)}',flush=True)
    manifest=dict(status='COMPLETE_NOT_FROZEN' if all(r['n_bootstrap_failed']==0 for r in temperature_rows) else 'OPTIMIZER_FAILURES_REVIEW_REQUIRED',
        result_freeze=False,replicates=2000,seed=20260930,input_sha256=input_hashes,
        notes='Point estimates reproduced. Pool labels only for T; test labels never enter fit. Metric intervals conditional on T; H4 descriptive, not Holm-tested.')
    manifest['output_sha256']={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')


if __name__=='__main__':main()
