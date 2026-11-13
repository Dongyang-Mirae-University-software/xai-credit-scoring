"""
보너스 과제 5: A/B 테스트 시뮬레이션.

과제 노션 원문
- 기존 CB 모델과 대안신용평가 모델의 승인 결과를 비교하는 A/B 테스트 시뮬레이션을 구현하고, 예상 대손율 변화를 분석한다.

용어
- A (기존 CB 모델): 금융 칸만 쓰는 전통 모델. B (대안신용평가 모델): 금융 + 대안 칸을 쓰는 통합 모델.
- 승인율 = 승인 고객 ÷ 신청 고객. 대손율 = 승인 고객 중 실제 부도(SeriousDlqin2yrs = 1) 비율.
  실제 대손율은 부도 금액·회수율까지 보지만, 이 데이터에는 금액이 없어 승인 고객의 부도율로 대신한다.

비교 방법
1. 정책 비교 (모든 고객에게 두 모델을 다 적용, 시뮬레이션에서만 가능)
   - 같은 기준값: 두 모델 모두 부도 확률 < 기준값이면 승인
   - 같은 승인율: B 의 기준값을 A 와 승인율이 같도록 맞춤 → 대손율이 얼마나 줄어드나
   - 같은 대손율: B 의 기준값을 A 와 대손율이 같도록 맞춤 → 승인을 얼마나 더 할 수 있나
2. A/B 테스트 (실제 실험처럼 고객을 무작위로 반씩 나눠 한쪽은 A, 한쪽은 B 로 심사)
   - 두 비율 z 검정으로 승인율·대손율 차이가 우연인지 확인, 나누기를 여러 번 반복해 결과가 흔들리는 폭을 본다.
"""

import numpy as np
from scipy import stats


def approval_metrics(y, p, threshold, mask=None):
    """기준값 하나에서 승인율·대손율. mask 를 주면 그 고객만."""
    y, p = np.asarray(y), np.asarray(p)
    if mask is not None:
        y, p = y[mask], p[mask]
    approved = p < threshold
    n_app = int(approved.sum())
    bad_app = int(y[approved].sum())
    return {
        'n': int(len(y)), 'n_approved': n_app, 'approval_rate': n_app / len(y) if len(y) else 0.0,
        'n_bad_approved': bad_app, 'bad_rate': bad_app / n_app if n_app else 0.0,
        'bad_caught': int(y[~approved].sum()), 'n_bad': int(y.sum()),
    }


def threshold_for_approval_rate(p, rate):
    """승인율이 rate 가 되는 기준값 (부도 확률이 낮은 순으로 rate 만큼 승인)."""
    p = np.sort(np.asarray(p))
    k = int(round(rate * len(p)))
    if k <= 0:
        return float(p[0])
    if k >= len(p):
        return float(np.nextafter(p[-1], np.inf))
    return float((p[k - 1] + p[k]) / 2) if p[k] > p[k - 1] else float(p[k])


def threshold_for_bad_rate(y, p, target):
    """대손율이 target 이하인 범위에서 가장 많이 승인하는 기준값."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind='stable')
    ys, ps = y[order], p[order]
    cum_bad = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    rate = cum_bad / k
    # 같은 확률 값이 여러 개면 그 묶음 끝에서만 자를 수 있다
    cut_ok = np.r_[ps[1:] > ps[:-1], True]
    ok = np.where((rate <= target) & cut_ok)[0]
    if len(ok) == 0:
        return float(ps[0])
    i = ok[-1]
    return float(np.nextafter(ps[i], np.inf))


def two_proportion_test(x1, n1, x2, n2):
    """두 비율 차이(2 - 1)의 z 검정과 95% 구간."""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se0 if se0 > 0 else 0.0
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {'rate_a': float(p1), 'rate_b': float(p2), 'diff': float(p2 - p1),
            'ci_low': float(p2 - p1 - 1.96 * se), 'ci_high': float(p2 - p1 + 1.96 * se),
            'z': float(z), 'p_value': float(2 * stats.norm.sf(abs(z)))}


def policy_comparison(y, p_a, p_b, threshold=0.5, segment=None):
    """세 정책(같은 기준값 / 같은 승인율 / 같은 대손율)에서 A, B 의 결과와 세그먼트별 결과."""
    a = approval_metrics(y, p_a, threshold)
    thr = {
        'same_threshold': threshold,
        'same_approval_rate': threshold_for_approval_rate(p_b, a['approval_rate']),
        'same_bad_rate': threshold_for_bad_rate(y, p_b, a['bad_rate']),
    }
    out = {'threshold_a': threshold, 'a': a, 'policies': {}}
    for name, t in thr.items():
        b = approval_metrics(y, p_b, t)
        row = {'threshold_b': float(t), 'b': b,
               'approval_change': b['approval_rate'] - a['approval_rate'],
               'bad_rate_change': b['bad_rate'] - a['bad_rate'],
               'bad_rate_relative_change': (b['bad_rate'] - a['bad_rate']) / a['bad_rate'] if a['bad_rate'] else 0.0}
        if segment is not None:
            seg = np.asarray(segment, dtype=bool)
            row['segments'] = {
                name_s: {'a': approval_metrics(y, p_a, threshold, m), 'b': approval_metrics(y, p_b, t, m)}
                for name_s, m in (('씬파일러', seg), ('일반', ~seg))
            }
        out['policies'][name] = row
    return out


def ab_split(y, p_a, p_b, thr_a, thr_b, seed=42, share_b=0.5):
    """고객을 무작위로 A/B 에 나눠 각 쪽을 그 모델로 심사한 결과와 차이 검정."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    to_b = rng.random(len(y)) < share_b
    a = approval_metrics(y[~to_b], np.asarray(p_a)[~to_b], thr_a)
    b = approval_metrics(y[to_b], np.asarray(p_b)[to_b], thr_b)
    return {
        'a': a, 'b': b,
        'approval_test': two_proportion_test(a['n_approved'], a['n'], b['n_approved'], b['n']),
        'bad_rate_test': two_proportion_test(a['n_bad_approved'], a['n_approved'], b['n_bad_approved'], b['n_approved']),
    }


def repeated_ab(y, p_a, p_b, thr_a, thr_b, n_rep=1000, seed=42):
    """A/B 나누기를 n_rep 번 반복: 대손율·승인율 차이의 분포와 '유의하게 나온 비율'(검정력)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_rep)
    d_bad, d_app, sig_bad = [], [], 0
    for s in seeds:
        r = ab_split(y, p_a, p_b, thr_a, thr_b, seed=int(s))
        d_bad.append(r['bad_rate_test']['diff'])
        d_app.append(r['approval_test']['diff'])
        sig_bad += r['bad_rate_test']['p_value'] < 0.05
    d_bad, d_app = np.array(d_bad), np.array(d_app)
    return {
        'n_rep': n_rep,
        'bad_rate_diff': {'mean': float(d_bad.mean()), 'p2_5': float(np.percentile(d_bad, 2.5)),
                          'p97_5': float(np.percentile(d_bad, 97.5))},
        'approval_diff': {'mean': float(d_app.mean()), 'p2_5': float(np.percentile(d_app, 2.5)),
                          'p97_5': float(np.percentile(d_app, 97.5))},
        'share_significant_bad_rate': sig_bad / n_rep,
        'bad_rate_diffs': d_bad.tolist(),
    }


def required_sample_size(rate_a, rate_b, approval_rate, alpha=0.05, power=0.8):
    """
    대손율 차이(rate_a → rate_b)를 검정력 power 로 찾아내려면 한쪽에 필요한 승인 고객 수와 신청 고객 수.
    두 비율 비교의 정규 근사: n = (z_α/2 + z_β)² × (p1(1-p1) + p2(1-p2)) ÷ (p1 - p2)²
    """
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    n_app = z ** 2 * (rate_a * (1 - rate_a) + rate_b * (1 - rate_b)) / (rate_a - rate_b) ** 2
    n_app = int(np.ceil(n_app))
    return {'approved_per_arm': n_app, 'applicants_per_arm': int(np.ceil(n_app / approval_rate)),
            'applicants_total': int(np.ceil(n_app / approval_rate)) * 2, 'alpha': alpha, 'power': power}


def strategy_curve(y, p, rates=None, mask=None):
    """승인율을 바꿔 가며 대손율 (승인율 → 대손율 곡선)."""
    y, p = np.asarray(y), np.asarray(p)
    if mask is not None:
        y, p = y[mask], p[mask]
    order = np.argsort(p, kind='stable')
    cum = np.cumsum(y[order])
    rates = np.linspace(0.3, 1.0, 71) if rates is None else np.asarray(rates)
    k = np.maximum(1, np.round(rates * len(y)).astype(int))
    return {'approval_rate': rates.tolist(), 'bad_rate': (cum[k - 1] / k).tolist()}
