"""
보너스 과제 3: Two-way ANOVA (심화 통계 검증).

과제 노션 원문
- 고객 세그먼트(씬파일러/일반) × 모델 구성(전통/통합)에 대한 Two-way ANOVA를 수행하고,
  교호작용(Interaction Effect)을 분석한다.

관측값: 교차검증 폴드마다, 같은 폴드의 검증 고객을 세그먼트로 나눠 잰 AUC.
  한 폴드에서 칸 4개(씬파일러/일반 × 전통/통합)가 하나씩 나온다 → 칸마다 폴드 수만큼 (균형 설계).
교호작용이 유의하면 "대안 데이터를 넣었을 때 AUC가 오르는 폭이 세그먼트마다 다르다"는 뜻이다.

폴드 안의 칸 4개는 같은 고객·같은 분할을 공유하므로 서로 독립이 아니다.
그래서 폴드별 차이의 차이(difference-in-differences)로 대응 t 검정도 함께 계산해 결론을 확인한다.
"""

import itertools

import numpy as np
from scipy import stats

SEGMENTS = ('씬파일러', '일반')
CONFIGS = ('전통', '통합')
CONFIG_DATA_TYPE = {'전통': 'financial_only', '통합': 'integrated'}


def _eta_label(eta):
    if eta >= 0.14:
        return '큼'
    if eta >= 0.06:
        return '중간'
    if eta >= 0.01:
        return '작음'
    return '거의 없음'


def two_way_anova(records, factor_a='segment', factor_b='config', value='auc'):
    """
    균형 설계 Two-way ANOVA (교호작용 포함). records: [{factor_a: .., factor_b: .., value: ..}, ...]
    돌려주는 값: 요인별 SS, df, F, p, η², 부분 η² 와 칸 평균.
    """
    levels_a = sorted({r[factor_a] for r in records}, key=str)
    levels_b = sorted({r[factor_b] for r in records}, key=str)
    cells = {(a, b): np.array([r[value] for r in records if r[factor_a] == a and r[factor_b] == b], dtype=float)
             for a, b in itertools.product(levels_a, levels_b)}
    sizes = {len(v) for v in cells.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError(f'칸마다 관측 수가 같아야 합니다 (균형 설계): {sorted(sizes)}')
    n = sizes.pop()
    if n < 2:
        raise ValueError('칸마다 관측이 2개 이상이어야 오차 분산을 계산할 수 있습니다.')
    ka, kb = len(levels_a), len(levels_b)
    allv = np.concatenate(list(cells.values()))
    grand = allv.mean()
    mean_a = {a: np.concatenate([cells[(a, b)] for b in levels_b]).mean() for a in levels_a}
    mean_b = {b: np.concatenate([cells[(a, b)] for a in levels_a]).mean() for b in levels_b}
    cell_mean = {k: v.mean() for k, v in cells.items()}

    ss_a = kb * n * sum((m - grand) ** 2 for m in mean_a.values())
    ss_b = ka * n * sum((m - grand) ** 2 for m in mean_b.values())
    ss_cells = n * sum((m - grand) ** 2 for m in cell_mean.values())
    ss_ab = ss_cells - ss_a - ss_b
    ss_e = sum(((v - v.mean()) ** 2).sum() for v in cells.values())
    ss_t = ((allv - grand) ** 2).sum()
    df_a, df_b, df_ab, df_e = ka - 1, kb - 1, (ka - 1) * (kb - 1), ka * kb * (n - 1)
    ms_e = ss_e / df_e

    def row(name, ss, df):
        f = (ss / df) / ms_e if ms_e > 0 else float('inf')
        eta = ss / ss_t if ss_t > 0 else 0.0
        return {'source': name, 'ss': float(ss), 'df': int(df), 'f_statistic': float(f),
                'p_value': float(stats.f.sf(f, df, df_e)), 'eta_squared': float(eta),
                'partial_eta_squared': float(ss / (ss + ss_e)) if ss + ss_e > 0 else 0.0,
                'effect_size': _eta_label(eta), 'is_significant': bool(stats.f.sf(f, df, df_e) < 0.05)}

    table = [row(factor_a, ss_a, df_a), row(factor_b, ss_b, df_b), row(f'{factor_a}×{factor_b}', ss_ab, df_ab)]
    table.append({'source': 'residual', 'ss': float(ss_e), 'df': int(df_e)})
    return {
        'table': table,
        'n_per_cell': n,
        'cell_means': {f'{a}|{b}': float(m) for (a, b), m in cell_mean.items()},
        'cell_std': {f'{a}|{b}': float(v.std(ddof=1)) for (a, b), v in cells.items()},
        'main_means': {factor_a: {str(k): float(v) for k, v in mean_a.items()},
                       factor_b: {str(k): float(v) for k, v in mean_b.items()}},
        'ss_total': float(ss_t),
    }


def paired_checks(records, segments=SEGMENTS, configs=CONFIGS):
    """
    폴드를 짝으로 묶은 확인. 폴드마다
      uplift[세그먼트] = AUC(통합) - AUC(전통),  did = uplift[씬파일러] - uplift[일반]
    을 구해 대응 t 검정(평균이 0인가)을 한다. 단순 주효과(세그먼트별 uplift)도 함께 돌려준다.
    """
    folds = sorted({r['fold_id'] for r in records})
    get = {(r['fold_id'], r['segment'], r['config']): r['auc'] for r in records}
    thin, normal = segments
    trad, integ = configs
    uplift = {s: np.array([get[(f, s, integ)] - get[(f, s, trad)] for f in folds]) for s in segments}
    did = uplift[thin] - uplift[normal]

    def ttest(x):
        t, p = stats.ttest_1samp(x, 0.0)
        half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return {'mean': float(x.mean()), 'ci_low': float(x.mean() - half), 'ci_high': float(x.mean() + half),
                't_statistic': float(t), 'p_value': float(p), 'n_folds': int(len(x)),
                'positive_folds': int((x > 0).sum())}

    return {
        'simple_effects': {s: ttest(uplift[s]) for s in segments},
        'interaction_did': ttest(did),
        'per_fold': [{'fold_id': f, **{f'uplift_{s}': float(uplift[s][i]) for s in segments}, 'did': float(did[i])}
                     for i, f in enumerate(folds)],
    }


def fold_auc_records(X, y, segment_mask, templates, feature_selector, n_splits=5, n_repeats=2, seed=42):
    """
    반복 층화 K-Fold 로 칸별 AUC 관측값을 만든다.
    templates: {'전통': 모델, '통합': 모델} (clone 해서 폴드마다 다시 학습)
    feature_selector(X, data_type) → 그 구성이 쓰는 칸만 남긴 X
    segment_mask: True = 씬파일러
    """
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    y = np.asarray(y)
    thin = np.asarray(segment_mask, dtype=bool)
    # 층을 (부도 여부, 씬파일러) 조합으로 나눠 폴드마다 씬파일러 부도가 고르게 들어가게 한다
    strata = y * 2 + thin.astype(int)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    Xs = {c: feature_selector(X, CONFIG_DATA_TYPE[c]) for c in templates}
    records = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), strata)):
        for config, template in templates.items():
            model = clone(template).fit(Xs[config].iloc[tr], y[tr])
            p = model.predict_proba(Xs[config].iloc[te])[:, 1]
            for seg_name, mask in ((SEGMENTS[0], thin[te]), (SEGMENTS[1], ~thin[te])):
                records.append({'fold_id': k, 'repeat': k // n_splits, 'segment': seg_name, 'config': config,
                                'auc': float(roc_auc_score(y[te][mask], p[mask])),
                                'n': int(mask.sum()), 'n_default': int(y[te][mask].sum())})
    return records


def interpret(anova, paired):
    """보고서·API용 해석 문장 (한국어)."""
    rows = {r['source']: r for r in anova['table']}
    inter = rows['segment×config']
    cm = anova['cell_means']
    s = []
    for name, key in (('고객 세그먼트', 'segment'), ('모델 구성', 'config')):
        r = rows[key]
        s.append(f"{name} 주효과: F = {r['f_statistic']:.2f}, p = {r['p_value']:.3g}, η² = {r['eta_squared']:.3f} ({r['effect_size']})")
    s.append(f"교호작용: F = {inter['f_statistic']:.2f}, p = {inter['p_value']:.3g}, "
             f"η² = {inter['eta_squared']:.3f}, 부분 η² = {inter['partial_eta_squared']:.3f}")
    up = paired['simple_effects']
    s.append(f"대안 데이터를 넣었을 때 AUC 향상: 씬파일러 {up['씬파일러']['mean']:+.3f} "
             f"(전통 {cm['씬파일러|전통']:.3f} → 통합 {cm['씬파일러|통합']:.3f}), "
             f"일반 {up['일반']['mean']:+.3f} (전통 {cm['일반|전통']:.3f} → 통합 {cm['일반|통합']:.3f})")
    did = paired['interaction_did']
    if inter['is_significant']:
        bigger = '씬파일러' if did['mean'] > 0 else '일반 고객'
        s.append(f"교호작용이 유의하다: 대안 데이터의 효과가 {bigger}에서 더 크다 "
                 f"(향상 차이 {did['mean']:+.3f}, 폴드 대응 t 검정 p = {did['p_value']:.3g}).")
    else:
        s.append(f"교호작용은 유의하지 않다: 대안 데이터의 효과가 두 세그먼트에서 비슷하다 "
                 f"(향상 차이 {did['mean']:+.3f}, 폴드 대응 t 검정 p = {did['p_value']:.3g}).")
    return s
