"""
ANOVA 통계 검증: 피처 그룹별·알고리즘별 One-way ANOVA, 효과 크기 η², 사후 검정 Tukey HSD.

원본 과제 기준
- 「`src/statistics/anova.py` 파일이 존재한다.」
- 「다음 2가지 ANOVA 검증을 필수로 수행한다」
  검증 1: One-way ANOVA, 피처 그룹별 모델 (전통 Only / 대안 Only / 통합), 대안 데이터 효용성 검증
  검증 2: One-way ANOVA, 알고리즘별 (LR / XGB / LGBM), 최적 알고리즘 선정 근거
- 「각 검증 결과에 F-통계량, p-value, η²(효과 크기)가 포함된다.」
- 「p < 0.05인 경우 사후 검정(Tukey HSD) 결과가 포함된다.」
- 「`/anova` API 호출 시 검증 결과가 JSON으로 반환된다.」
- 「리포트에 ANOVA 결과 해석이 1페이지 이상 포함되어 있다.」
- η² 해석 기준: 0.01-0.06 Small, 0.06-0.14 Medium, 0.14 이상 Large

재료: 학습할 때 MLflow 에 기록되는 폴드별 AUC (fold_1_roc_auc ... fold_5_roc_auc).
출력 형식: 대시보드 mock(src/dashboard/mock/statistical_test/anova.json)과 같은 키.
유의수준은 환경 변수 ANOVA_ALPHA(기본 0.05)로 바꿀 수 있다.
"""

import itertools
import os

DISCLAIMER = "본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."
ALPHA = float(os.getenv('ANOVA_ALPHA', '0.05'))
N_FOLDS = int(os.getenv('ANOVA_N_FOLDS', '5'))

# MLflow params 값 → 대시보드 표시 이름 (mock 과 같은 이름)
DATA_TYPE_LABELS = {
    'financial_only': '전통모델',
    'alternative_only': '대안모델',
    'integrated': '통합모델',
}
ALGORITHM_LABELS = {
    'Logistic Regression': 'LogisticRegression',
    'LogisticRegression': 'LogisticRegression',
    'LR': 'LogisticRegression',
    'XGBoost': 'XGBoost',
    'XGB': 'XGBoost',
    'LightGBM': 'LightGBM',
    'LGBM': 'LightGBM',
}

TEST_1_NAME = "피처 그룹별 AUC 비교 (One-way ANOVA)"
TEST_2_NAME = "알고리즘별 AUC 비교 (One-way ANOVA)"


def interpret_eta_squared(eta):
    """원본 η² 해석 기준."""
    if eta >= 0.14:
        return 'Large'
    if eta >= 0.06:
        return 'Medium'
    if eta >= 0.01:
        return 'Small'
    return 'Negligible'


def eta_squared(groups):
    """η² = 집단 간 제곱합 / 전체 제곱합."""
    values = [v for g in groups for v in g]
    grand = sum(values) / len(values)
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ss_total = sum((v - grand) ** 2 for v in values)
    return ss_between / ss_total if ss_total > 0 else 0.0


def _short(label):
    return label[:-2] if label.endswith('모델') else label


def one_way_anova(groups, test_name='One-way ANOVA', alpha=None):
    """groups: {집단 이름: [점수, ...]} (순서 유지). F, p, η², 유의하면 Tukey HSD."""
    from scipy import stats
    alpha = ALPHA if alpha is None else alpha
    names = list(groups)
    data = [list(map(float, groups[n])) for n in names]
    if len(data) < 2:
        raise ValueError("ANOVA 에는 집단이 2개 이상 필요합니다")
    for n, d in zip(names, data):
        if len(d) < 2:
            raise ValueError(f"집단 '{n}' 의 점수가 2개 미만입니다")
    f, p = stats.f_oneway(*data)
    eta = eta_squared(data)
    result = {
        'test_name': test_name,
        'groups': names,
        'group_means': {n: sum(d) / len(d) for n, d in zip(names, data)},
        'group_sizes': {n: len(d) for n, d in zip(names, data)},
        'f_statistic': float(f),
        'p_value': float(p),
        'effect_size_eta_squared': float(eta),
        'effect_size_interpretation': interpret_eta_squared(eta),
        'alpha': alpha,
        'is_significant': bool(p < alpha),
        'post_hoc_tukey': [],
    }
    if result['is_significant']:
        tukey = stats.tukey_hsd(*data)
        for i, j in itertools.combinations(range(len(names)), 2):
            p_adj = float(tukey.pvalue[i, j])
            result['post_hoc_tukey'].append({
                'comparison': f"{_short(names[i])} vs {_short(names[j])}",
                'mean_diff': result['group_means'][names[i]] - result['group_means'][names[j]],
                'p_adj': p_adj,
                'significant': bool(p_adj < alpha),
            })
    return result


# ---------- MLflow 기록에서 폴드 점수 모으기 ----------

def fold_scores(metrics, n_folds=None, metric='roc_auc'):
    """run 의 metrics 에서 fold_1_<metric> ... fold_k_<metric> 를 순서대로 꺼낸다. 하나라도 없으면 None."""
    n = N_FOLDS if n_folds is None else n_folds
    keys = [f'fold_{k}_{metric}' for k in range(1, n + 1)]
    if not all(k in metrics for k in keys):
        return None
    return [float(metrics[k]) for k in keys]


def latest_per_combo(records):
    """같은 (algorithm, method, data_type) 조합이 여러 번 기록됐으면 가장 최근 것만 남긴다.
    records: [{'params':{}, 'metrics':{}, 'status':'FINISHED', 'start_time': ms}, ...]"""
    best = {}
    for r in records:
        if r.get('status', 'FINISHED') != 'FINISHED':
            continue
        p = r['params']
        key = (p.get('algorithm'), p.get('method'), p.get('data_type'))
        if None in key:
            continue
        if key not in best or r.get('start_time', 0) > best[key].get('start_time', 0):
            best[key] = r
    return list(best.values())


def group_scores(records, by, labels, where=None, metric='roc_auc'):
    """by(params 이름)별로 폴드 점수를 모은다. where: {param: 값} 조건. 표시 이름 순서는 labels 순서."""
    groups = {}
    for r in latest_per_combo(records):
        p = r['params']
        if where and any(p.get(k) != v for k, v in where.items()):
            continue
        label = labels.get(p.get(by))
        scores = fold_scores(r['metrics'], metric=metric)
        if label is None or scores is None:
            continue
        groups.setdefault(label, []).extend(scores)
    order = list(dict.fromkeys(labels.values()))
    return {k: groups[k] for k in order if k in groups}


def run_anova_feature_group(records, metric='roc_auc'):
    """검증 1: 피처 그룹별 모델(전통/대안/통합) One-way ANOVA. (초기 구조의 함수 이름 유지)"""
    return one_way_anova(group_scores(records, 'data_type', DATA_TYPE_LABELS, metric=metric), TEST_1_NAME)


def run_anova_algorithm(records, data_type='integrated', metric='roc_auc'):
    """검증 2: 알고리즘별(LR/XGB/LGBM) One-way ANOVA. 기본은 통합 데이터 모델끼리 비교. (초기 구조의 함수 이름 유지)"""
    groups = group_scores(records, 'algorithm', ALGORITHM_LABELS, where={'data_type': data_type}, metric=metric)
    return one_way_anova(groups, TEST_2_NAME)


def anova_report(records, test2_data_type='integrated', metric='roc_auc'):
    """/anova 응답 형식 (mock 과 같은 test_1 / test_2 키)."""
    report = {
        'test_1': run_anova_feature_group(records, metric=metric),
        'test_2': run_anova_algorithm(records, data_type=test2_data_type, metric=metric),
        'source': {
            'metric': f'fold_k_{metric}',
            'runs_used': len(latest_per_combo(records)),
            'test_2_data_type': test2_data_type,
        },
        'disclaimer': DISCLAIMER,
    }
    report['interpretation'] = interpret_report(report)
    return report


# ---------- 리포트용 해석 문장 ----------

def _has_batchim(word):
    """마지막 글자에 받침이 있는지. 영어 이름은 읽는 소리 기준(…L·M·N·R 로 끝나면 받침)."""
    ch = word.strip()[-1]
    if '가' <= ch <= '힣':
        return (ord(ch) - ord('가')) % 28 != 0
    return ch.lower() in 'lmnr'


def _josa(word, with_batchim, without_batchim):
    return word + (with_batchim if _has_batchim(word) else without_batchim)


def interpret(result):
    """원본 해석 예시 형식의 문장들."""
    means = ', '.join(f"{n}(M={m:.3f})" for n, m in result['group_means'].items())
    lines = [
        f"{result['test_name']}: {means}.",
        f"F = {result['f_statistic']:.2f}, p = {result['p_value']:.3g}, "
        f"η² = {result['effect_size_eta_squared']:.3f} ({result['effect_size_interpretation']}).",
    ]
    if not result['is_significant']:
        lines.append(f"p ≥ {result['alpha']} 이므로 집단 간 평균 AUC 차이는 통계적으로 유의하지 않다.")
        return lines
    lines.append(f"p < {result['alpha']} 이므로 집단 간 평균 AUC 차이가 통계적으로 유의하여 Tukey HSD 사후 검정을 수행했다.")
    for t in result['post_hoc_tukey']:
        a, b = t['comparison'].split(' vs ')
        if t['significant']:
            hi, lo = (a, b) if t['mean_diff'] > 0 else (b, a)
            lines.append(f"{_josa(hi, '이', '가')} {lo}보다 유의하게 높은 AUC를 보였다 (차이 {abs(t['mean_diff']):.3f}, p_adj={t['p_adj']:.3g}).")
        else:
            lines.append(f"{_josa(a, '과', '와')} {b} 간에는 유의한 차이가 없었다 (p_adj={t['p_adj']:.3g}).")
    return lines


def interpret_report(report):
    return {k: interpret(report[k]) for k in ('test_1', 'test_2')}


# ---------- 기록 불러오기 ----------

def records_from_rest(payload):
    """MLflow REST runs/search 응답(JSON) → records. 서버에 직접 붙지 않고 내보낸 파일로 계산할 때 쓴다."""
    out = []
    for r in payload.get('runs', []):
        info, data = r['info'], r.get('data', {})
        out.append({
            'run_name': info.get('run_name'),
            'status': info.get('status'),
            'start_time': int(info.get('start_time', 0)),
            'params': {x['key']: x['value'] for x in data.get('params', [])},
            'metrics': {x['key']: x['value'] for x in data.get('metrics', [])},
        })
    return out


def records_from_mlflow(experiment_name=None, max_results=1000):
    """MLflow 서버에서 실험 기록을 읽는다. 주소·계정은 환경 변수(MLFLOW_TRACKING_URI 등)로 정한다."""
    from mlflow.tracking import MlflowClient
    name = experiment_name or os.getenv('MLFLOW_EXPERIMENT_NAME', 'xai-credit-scoring')
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise ValueError(f"MLflow 실험 '{name}' 을 찾을 수 없습니다")
    runs = client.search_runs([exp.experiment_id], max_results=max_results)
    return [{
        'run_name': r.info.run_name,
        'status': r.info.status,
        'start_time': int(r.info.start_time or 0),
        'params': dict(r.data.params),
        'metrics': dict(r.data.metrics),
    } for r in runs]
