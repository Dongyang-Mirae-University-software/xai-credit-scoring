"""
보너스 과제 4: 모델 드리프트 탐지 (PSI + CSI).

과제 노션 원문
- PSI와 함께 CSI(Characteristic Stability Index)를 계산하여 개별 피처 수준의 분포 변화를 탐지하는 모듈을 구현한다.

PSI 는 모델 점수(부도 확률) 하나의 분포 변화, CSI 는 입력 칸(피처) 하나하나의 분포 변화를 같은 식으로 잰다.
  지표 = Σ (비교 비율 - 기준 비율) × ln(비교 비율 / 기준 비율)   (구간마다 더함)
구간은 기준(보통 Train) 분포로 정한다.
  - 값 종류가 구간 수 이하인 칸(정기결제 건수, 앱 로그인 빈도 등)은 값 하나를 한 구간으로 쓰고,
    기준 범위 밖(최솟값 미만 / 최댓값 초과) 구간을 따로 둔다. 새 값이 들어오는 것도 드리프트이기 때문이다.
  - 연속 칸은 기준 분포의 분위수로 나누고, 같은 값이 많아 겹치는 경계는 합친다.
판정: 0.1 미만 안정, 0.1 ~ 0.25 주의, 0.25 이상 경고 (업계에서 흔히 쓰는 기준, PSI 목표 < 0.1 과 같은 선).

PSI 만 보면 "무엇이 바뀌었나"를 모른다. CSI 는 어느 칸의 어느 구간에서 비율이 바뀌었는지까지 보여 줘서
재학습이 필요한지, 데이터 수집에 문제가 생겼는지 가려내는 데 쓴다.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

STABLE, CAUTION = 0.1, 0.25
EPS = 1e-6


def grade(value):
    """지표 값 → '안정' / '주의' / '경고'."""
    if value < STABLE:
        return '안정'
    if value < CAUTION:
        return '주의'
    return '경고'


def bin_edges(expected, bins=10):
    """
    기준 값으로 구간을 정한다. 돌려주는 값: (종류, 경계)
      ('discrete', 정렬된 고유값)  값 종류가 bins 이하
      ('quantile', 내부 경계)      그 외. 구간 i 는 (경계[i-1], 경계[i]] (오른쪽 닫힘)
    """
    exp = np.asarray(expected, dtype=float)
    exp = exp[~np.isnan(exp)]
    if exp.size == 0:
        raise ValueError('기준 값이 비어 있습니다.')
    uniq = np.unique(exp)
    if uniq.size <= bins:
        return 'discrete', uniq
    edges = np.unique(np.quantile(exp, np.arange(1, bins) / bins))
    return 'quantile', edges


def _assign(values, kind, edges):
    """
    discrete: 0 = 기준 최솟값보다 작음, 1..k = 기준 값(사이 값은 가까운 아래 값), k+1 = 기준 최댓값보다 큼
    quantile: 0..len(edges)
    """
    v = np.asarray(values, dtype=float)
    if kind == 'discrete':
        idx = np.searchsorted(edges, v, side='right')          # 1..k (최솟값보다 작으면 0)
        idx[v > edges[-1]] = len(edges) + 1
        return idx
    return np.searchsorted(edges, v, side='left')


def _n_bins(kind, edges):
    return len(edges) + 2 if kind == 'discrete' else len(edges) + 1


def _labels(kind, edges):
    if kind == 'discrete':
        return [f'{edges[0]:g} 미만 (기준 범위 밖)'] + [f'{e:g}' for e in edges] + [f'{edges[-1]:g} 초과 (기준 범위 밖)']
    lo = ['최소'] + [f'{e:.4g}' for e in edges]
    hi = [f'{e:.4g}' for e in edges] + ['최대']
    return [f'{a} ~ {b}' for a, b in zip(lo, hi)]


def stability_table(expected, actual, bins=10):
    """
    구간별 기준 비율 · 비교 비율 · 기여도와 합계(지표 값).
    돌려주는 값: {'index': float, 'grade': str, 'bins': [{label, expected, actual, contribution}, ...]}
    """
    kind, edges = bin_edges(expected, bins)
    n_bins = _n_bins(kind, edges)
    e = np.bincount(_assign(expected, kind, edges), minlength=n_bins) / len(np.asarray(expected))
    a = np.bincount(_assign(actual, kind, edges), minlength=n_bins) / len(np.asarray(actual))
    contrib = (a - e) * np.log((a + EPS) / (e + EPS))
    total = float(contrib.sum())
    rows = [{'label': lab, 'expected': float(ei), 'actual': float(ai), 'contribution': float(ci)}
            for lab, ei, ai, ci in zip(_labels(kind, edges), e, a, contrib)]
    return {'index': total, 'grade': grade(total), 'binning': kind, 'bins': rows}


def psi(expected_scores, actual_scores, bins=10):
    """모델 점수(부도 확률)의 PSI."""
    return stability_table(expected_scores, actual_scores, bins)['index']


def csi(expected_df, actual_df, columns, bins=10):
    """칸별 CSI. {칸: {'index', 'grade', 'binning', 'bins'}}"""
    return {c: stability_table(expected_df[c], actual_df[c], bins) for c in columns}


def drift_report(expected_df, actual_df, columns, expected_scores=None, actual_scores=None, bins=10, name='비교'):
    """
    PSI(점수) + CSI(칸별) 를 한 번에. 주의 이상인 칸은 WARNING 로그를 남긴다.
    돌려주는 값: score_psi, score_grade, csi(칸별 값·판정), flagged(주의 이상, 큰 순), top_bins(칸별 기여도 큰 구간)
    """
    tables = csi(expected_df, actual_df, columns, bins)
    report = {'name': name, 'n_expected': int(len(expected_df)), 'n_actual': int(len(actual_df))}
    if expected_scores is not None and actual_scores is not None:
        score = stability_table(expected_scores, actual_scores, bins)
        report.update(score_psi=score['index'], score_grade=score['grade'], score_bins=score['bins'])
        if score['index'] >= STABLE:
            logger.warning('[%s] 점수 PSI %.4f (%s): 모델 점수 분포가 바뀌었습니다.', name, score['index'], score['grade'])
    report['csi'] = {c: {'index': t['index'], 'grade': t['grade']} for c, t in tables.items()}
    flagged = sorted((c for c, t in tables.items() if t['index'] >= STABLE), key=lambda c: -tables[c]['index'])
    report['flagged'] = flagged
    report['top_bins'] = {}
    for c in flagged:
        top = max(tables[c]['bins'], key=lambda b: b['contribution'])
        report['top_bins'][c] = top
        logger.warning('[%s] CSI %s = %.4f (%s), 가장 많이 바뀐 구간 %s: %.1f%% → %.1f%%', name, c, tables[c]['index'],
                       tables[c]['grade'], top['label'], top['expected'] * 100, top['actual'] * 100)
    report['tables'] = tables
    return report
