"""
모델 성능 비교: AUC-ROC, KS, PSI, 씬파일러 AUC 향상, ROC·PR 곡선, 혼동행렬.

원본 과제 기준
- Test Set AUC-ROC ≥ 0.78
- Test Set KS Statistic ≥ 0.28
- 씬파일러 집단 AUC 향상 +0.03 이상 (대안 데이터 포함 모델 vs 미포함 모델)
- PSI < 0.1 (Train/Test 분포 안정성)

numpy·sklearn 없이도 돌아가도록 파이썬 기본 기능으로 계산한다.
(리스트, numpy 배열, pandas Series 모두 입력으로 받을 수 있다.)
"""

import json
import math
import os
from pathlib import Path

DISCLAIMER = "본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."

# 원본 과제 기준값. 코드에 박지 않고 환경 변수로 바꿀 수 있게 한다.
TARGETS = {
    'auc': float(os.getenv('TARGET_AUC', '0.78')),
    'ks': float(os.getenv('TARGET_KS', '0.28')),
    'thin_filer_uplift': float(os.getenv('TARGET_THIN_UPLIFT', '0.03')),
    'psi': float(os.getenv('TARGET_PSI', '0.1')),
}


def _to_list(values):
    return [float(v) for v in values]


def _check(y_true, y_score):
    y = [int(v) for v in y_true]
    s = _to_list(y_score)
    if len(y) != len(s):
        raise ValueError(f"정답({len(y)}개)과 점수({len(s)}개)의 개수가 다릅니다.")
    if not any(y) or all(y):
        raise ValueError("정답에 부도(1)와 정상(0)이 모두 있어야 계산할 수 있습니다.")
    return y, s


def roc_auc(y_true, y_score):
    """AUC-ROC. 부도자에게 정상인보다 높은 점수를 줄 확률 (같으면 0.5로 셈)."""
    y, s = _check(y_true, y_score)
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    rank_sum = sum(r for r, t in zip(ranks, y) if t == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_curve(y_true, y_score):
    """ROC 곡선 점 목록. 반환: (fpr 리스트, tpr 리스트, threshold 리스트)."""
    y, s = _check(y_true, y_score)
    pairs = sorted(zip(s, y), key=lambda p: -p[0])
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    fpr, tpr, thr = [0.0], [0.0], [float('inf')]
    tp = fp = 0
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == score:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        fpr.append(fp / n_neg)
        tpr.append(tp / n_pos)
        thr.append(score)
    return fpr, tpr, thr


def ks_statistic(y_true, y_score):
    """KS 통계량. 부도자와 정상인의 누적 점수 분포가 가장 크게 벌어진 거리 (= max(TPR - FPR))."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return max(t - f for t, f in zip(tpr, fpr))


def precision_recall_curve(y_true, y_score):
    """PR 곡선 점 목록. 반환: (precision 리스트, recall 리스트, threshold 리스트)."""
    y, s = _check(y_true, y_score)
    pairs = sorted(zip(s, y), key=lambda p: -p[0])
    n_pos = sum(y)
    precision, recall, thr = [], [], []
    tp = fp = 0
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == score:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        precision.append(tp / (tp + fp))
        recall.append(tp / n_pos)
        thr.append(score)
    return precision, recall, thr


def confusion_matrix(y_true, y_score, threshold=0.5):
    """부도 확률이 threshold 이상이면 부도(1)로 본 혼동행렬과 정밀도·재현율·F1."""
    y, s = _check(y_true, y_score)
    tp = sum(1 for t, p in zip(y, s) if t == 1 and p >= threshold)
    fn = sum(1 for t, p in zip(y, s) if t == 1 and p < threshold)
    fp = sum(1 for t, p in zip(y, s) if t == 0 and p >= threshold)
    tn = sum(1 for t, p in zip(y, s) if t == 0 and p < threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'threshold': threshold,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
    }


def psi(expected_scores, actual_scores, bins=10):
    """
    PSI (Population Stability Index). 기준(보통 Train) 점수 분포를 10등분한 구간에
    비교 대상(보통 Test) 점수가 얼마나 비슷한 비율로 들어가는지 잰다. 0.1 미만이면 안정.
    """
    exp = sorted(_to_list(expected_scores))
    act = _to_list(actual_scores)
    if not exp or not act:
        raise ValueError("PSI 계산에는 두 점수 목록이 모두 필요합니다.")
    # 기준 분포의 분위수로 구간 경계를 정한다 (같은 값이 많으면 구간이 합쳐질 수 있음)
    edges = sorted({exp[min(len(exp) - 1, int(len(exp) * k / bins))] for k in range(1, bins)})

    def share(values):
        counts = [0] * (len(edges) + 1)
        for v in values:
            idx = 0
            while idx < len(edges) and v >= edges[idx]:
                idx += 1
            counts[idx] += 1
        return [c / len(values) for c in counts]

    eps = 1e-6
    e_share, a_share = share(exp), share(act)
    return sum((a - e) * math.log((a + eps) / (e + eps)) for e, a in zip(e_share, a_share))


def thin_filer_uplift(y_true, score_without_alt, score_with_alt, thin_mask):
    """
    씬파일러 집단 AUC 향상 = 씬파일러만 놓고 (대안 데이터 포함 모델 AUC - 미포함 모델 AUC).
    thin_mask: 사람마다 씬파일러면 1(True), 아니면 0(False).
    """
    y_all = [int(v) for v in y_true]
    s0_all = _to_list(score_without_alt)
    s1_all = _to_list(score_with_alt)
    idx = [i for i, m in enumerate(thin_mask) if int(m) == 1]
    if not idx:
        raise ValueError("씬파일러가 한 명도 없어 AUC 향상을 계산할 수 없습니다. is_thin_filer 칸을 확인하세요.")
    y = [y_all[i] for i in idx]
    s0 = [s0_all[i] for i in idx]
    s1 = [s1_all[i] for i in idx]
    auc_without = roc_auc(y, s0)
    auc_with = roc_auc(y, s1)
    return {
        'n_thin_filer': len(idx),
        'n_thin_filer_default': sum(y),
        'auc_without_alt': auc_without,
        'auc_with_alt': auc_with,
        'uplift': auc_with - auc_without,
    }


def evaluate(y_true, y_score, threshold=None, train_scores=None, curve_points=50):
    """
    한 모델의 Test 성능을 한 번에 계산해 dict 로 돌려준다 (API /metrics, 대시보드 '모델 성능' 탭 재료).
    threshold: 부도로 볼 확률 기준. 없으면 환경 변수 APPROVAL_THRESHOLD(기본 0.5).
    train_scores: 주면 PSI 도 계산한다.
    """
    if threshold is None:
        threshold = float(os.getenv('APPROVAL_THRESHOLD', '0.5'))
    y = [int(v) for v in y_true]
    s = _to_list(y_score)
    auc = roc_auc(y, s)
    ks = ks_statistic(y, s)
    cm = confusion_matrix(y, s, threshold)
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    result = {
        'auc': auc,
        'ks': ks,
        'precision': cm['precision'],
        'recall': cm['recall'],
        'f1': cm['f1'],
        'confusion_matrix': cm,
        'roc_curve': _thin_out({'fpr': fpr, 'tpr': tpr}, curve_points),
        'pr_curve': _thin_out({'precision': prec, 'recall': rec}, curve_points),
        'n': len(y),
        'n_default': sum(y),
        'pass': {
            'auc': auc >= TARGETS['auc'],
            'ks': ks >= TARGETS['ks'],
        },
        'targets': dict(TARGETS),
        'disclaimer': DISCLAIMER,
    }
    if train_scores is not None:
        value = psi(train_scores, s)
        result['psi'] = value
        result['pass']['psi'] = value < TARGETS['psi']
    return result


def _thin_out(curve, points):
    """그림용으로 곡선 점을 points 개 정도로 줄인다 (처음과 끝은 남김)."""
    keys = list(curve)
    n = len(curve[keys[0]])
    if n <= points:
        return curve
    step = (n - 1) / (points - 1)
    picks = sorted({round(i * step) for i in range(points)})
    return {k: [curve[k][i] for i in picks] for k in keys}


def save_json(result, path):
    """결과를 JSON 으로 저장한다 (한글 그대로)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path
