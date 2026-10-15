"""
MLflow에 남기는 쉬운 영어 설명.

키 이름(roc_auc_mean, bias_ratio 등)은 코드·검색·ANOVA 계산이 쓰므로 그대로 두고,
그 옆에 붙는 설명(run Description, 태그, 지표 안내 파일, 결과 요약)만 쉬운 영어로 만든다.
mlflow 패키지 없이 동작한다 (문자열만 만든다).
"""

import re

from src.explainer.performance import TARGETS as _PERF_TARGETS
from src.fairness.metrics import DI_MAX, DI_MIN, EO_MAX
from src.statistics.anova import ALPHA

DISCLAIMER_EN = "This system was built for education only and must not be used for real financial decisions."

# 기준값은 각 모듈의 환경 변수 설정을 그대로 따른다 (TARGET_*, FAIRNESS_*, ANOVA_ALPHA)
TARGETS = dict(_PERF_TARGETS, di_min=DI_MIN, di_max=DI_MAX, eo_max=EO_MAX, alpha=ALPHA)

DATA_TYPE_EN = {
    'financial_only': 'financial data only',
    'alternative_only': 'alternative data only',
    'integrated': 'financial + alternative data',
}
ALGORITHM_EN = {
    'LogisticRegression': 'Logistic Regression', 'Logistic Regression': 'Logistic Regression', 'LR': 'Logistic Regression',
    'XGBoost': 'XGBoost', 'XGB': 'XGBoost', 'LightGBM': 'LightGBM', 'LGBM': 'LightGBM',
}
_WEIGHT = 'give rare default cases more weight'
METHOD_EN = {
    'class_weight': _WEIGHT, "class_weight='balanced'": _WEIGHT, 'scale_pos_weight': _WEIGHT,
    'is_unbalance': _WEIGHT, 'smote': 'add synthetic default cases (SMOTE)', 'SMOTE': 'add synthetic default cases (SMOTE)',
}
FEATURE_EN = {
    'RevolvingUtilizationOfUnsecuredLines': 'credit card limit usage',
    'age': 'age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'late payments (30-59 days)',
    'DebtRatio': 'debt-to-income ratio',
    'MonthlyIncome': 'monthly income',
    'NumberOfOpenCreditLinesAndLoans': 'open loans and credit lines',
    'NumberOfTimes90DaysLate': 'late payments (90+ days)',
    'NumberRealEstateLoansOrLines': 'real estate loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'late payments (60-89 days)',
    'NumberOfDependents': 'number of dependents',
    'telecom_payment_rate': 'phone bill on-time rate',
    'utility_payment_rate': 'utility bill payment rate',
    'spending_consistency': 'spending consistency score',
    'regular_payment_count': 'number of regular payments',
    'app_login_frequency': 'app logins per month',
}
# ANOVA 집단 이름(src/statistics/anova.py 의 표시 이름) → 영어. 긴 이름부터 바꾼다.
GROUP_EN = {
    '전통모델': 'financial only', '대안모델': 'alternative only', '통합모델': 'financial + alternative',
    '전통': 'financial only', '대안': 'alternative only', '통합': 'financial + alternative',
}

BASE_METRICS = {
    'roc_auc': 'AUC: how well the model ranks people who default above people who repay. 0.5 = random guess, 1.0 = perfect.',
    'precision': 'Precision: of the people the model flagged as "will default", the share who really defaulted.',
    'recall': 'Recall: of the people who really defaulted, the share the model caught.',
    'f1': 'F1: one number that balances precision and recall. Higher is better.',
    'ks': 'KS: the biggest gap between the score curves of defaulters and repayers. Higher is better (goal: 0.28 or higher).',
}
EXACT_METRICS = {
    'selected_val_auc': 'AUC of the chosen model on the validation data (used only to pick the model).',
    'test_auc': 'AUC of the chosen model on test data it never saw. Goal: 0.78 or higher.',
    'test_ks': 'KS of the chosen model on test data. Goal: 0.28 or higher.',
    'test_psi': 'PSI of the model score: how much the score distribution moved between train and test. Below 0.1 = stable.',
    'feature_psi_max': 'The largest PSI among the 15 inputs. Below 0.1 = every input is stable.',
    'thin_uplift': 'Thin-filer AUC gain: AUC with alternative data minus AUC without it, for people with little credit history. Goal: +0.03 or more.',
    'thin_ci_low': 'Lower end of the 95% range for the thin-filer AUC gain (from 1,000 resamples).',
    'thin_ci_high': 'Upper end of the 95% range for the thin-filer AUC gain.',
    'thin_auc_without_alt': 'Thin-filer AUC of the best model that uses financial data only.',
    'thin_auc_with_alt': 'Thin-filer AUC of the best model that uses financial + alternative data.',
    'thin_n': 'Number of thin-filers in the test data.',
    'thin_n_default': 'Number of thin-filers in the test data who defaulted.',
}
PARAM_GUIDE = {
    'algorithm': 'Model type (Logistic Regression, XGBoost or LightGBM).',
    'method': 'How the model handles the small number of defaults (about 6.7% of people).',
    'data_type': 'Which inputs were used: financial only, alternative only, or both.',
    'thin_filer_ratio': 'Simulator setting for the share of thin-filers. None = use the team rule only.',
    'bias_ratio': 'Simulator setting that adds group bias for fairness tests. 0 = no added bias.',
    'n_estimators': 'Number of trees.',
    'max_depth': 'Maximum depth of each tree.',
    'learning_rate': 'How big each learning step is.',
    'max_iter': 'Maximum training steps for Logistic Regression.',
    'scale_pos_weight': 'Extra weight for default cases in XGBoost.',
    'selection_metric': 'The score used to pick the final model.',
    'uplift_model_without_alt': 'Model used as "without alternative data" in the thin-filer comparison.',
    'uplift_model_with_alt': 'Model used as "with alternative data" in the thin-filer comparison.',
}


def explain_key(key):
    """지표 키 → 쉬운 영어 한 문장. 모르는 키는 None."""
    if key in EXACT_METRICS:
        return EXACT_METRICS[key]
    if key.startswith('psi_'):
        name = key[4:]
        return f"PSI for one input ({FEATURE_EN.get(name, name)}): how much it moved between train and test. Below 0.1 = stable."
    m = re.fullmatch(r'fold_(\d+)_(\w+)', key)
    if m and m.group(2) in BASE_METRICS:
        return f"{BASE_METRICS[m.group(2)]} Measured on fold {m.group(1)} of 5."
    m = re.fullmatch(r'(\w+)_(mean|std)', key)
    if m and m.group(1) in BASE_METRICS:
        if m.group(2) == 'mean':
            return f"{BASE_METRICS[m.group(1)]} Average of the 5 folds."
        return f"How much the 5 fold scores differ ({m.group(1)}). Small = stable model."
    return BASE_METRICS.get(key)


def metrics_guide_md(metric_keys=None, param_keys=None):
    """지표·파라미터 이름표 (Markdown). 키를 안 주면 우리 프로젝트에서 쓰는 키 전체."""
    if metric_keys is None:
        metric_keys = (list(EXACT_METRICS) + [f'{b}_{s}' for b in BASE_METRICS for s in ('mean', 'std')]
                       + [f'fold_1_{b}' for b in BASE_METRICS] + [f'psi_{f}' for f in FEATURE_EN])
    if param_keys is None:
        param_keys = list(PARAM_GUIDE)
    lines = ['# What the names mean', '',
             'Names stay as they are in the code. This page explains them in plain English.', '',
             '## Metrics', '', '| Name | Meaning |', '|---|---|']
    for k in metric_keys:
        text = explain_key(k)
        if text:
            lines.append(f'| `{k}` | {text} |')
    lines += ['', '`fold_2_...` to `fold_5_...` mean the same as `fold_1_...` for the other folds.', '',
              '## Parameters', '', '| Name | Meaning |', '|---|---|']
    for k in param_keys:
        if k in PARAM_GUIDE:
            lines.append(f'| `{k}` | {PARAM_GUIDE[k]} |')
    lines += ['', f'> {DISCLAIMER_EN}', '']
    return '\n'.join(lines)


def _cv(cv_means, name):
    for key in (f'cv_test_{name}_mean', f'{name}_mean'):
        if key in cv_means:
            return float(cv_means[key])
    return None


def model_run_tags(algorithm, method, data_type):
    """모델 run에 붙일 쉬운 영어 태그."""
    return {
        'model_family': ALGORITHM_EN.get(algorithm, str(algorithm)),
        'data_used': DATA_TYPE_EN.get(data_type, str(data_type)),
        'imbalance_fix': METHOD_EN.get(method, str(method)),
    }


def model_run_note(algorithm, method, data_type, cv_means=None):
    """모델 run Description (쉬운 영어)."""
    tags = model_run_tags(algorithm, method, data_type)
    text = f"{tags['model_family']} model trained on {tags['data_used']}. Imbalance fix: {tags['imbalance_fix']}."
    cv_means = cv_means or {}
    auc, std = _cv(cv_means, 'roc_auc'), cv_means.get('roc_auc_std')
    if auc is not None:
        text += f" 5-fold average AUC: {auc:.3f}"
        text += f" (spread {float(std):.3f})." if std is not None else "."
    parts = [(n, _cv(cv_means, n)) for n in ('precision', 'recall', 'f1', 'ks')]
    parts = [f"{n} {v:.3f}" for n, v in parts if v is not None]
    if parts:
        text += " Other averages: " + ", ".join(parts) + "."
    return text


def _verdict(ok):
    return 'passed' if ok else 'NOT passed'


def final_test_note(selected_model, test_auc, test_ks, psi=None, uplift=None):
    """final_test run Description (쉬운 영어). uplift: {'uplift','ci_low','ci_high'} 또는 None."""
    data_type, _, rest = str(selected_model).partition('/')
    text = (f"The chosen model ({rest.replace('/', ', ') or selected_model}, {DATA_TYPE_EN.get(data_type, data_type)}) "
            f"was tested once on unseen test data. "
            f"AUC {test_auc:.3f} (goal {TARGETS['auc']} or higher, {_verdict(test_auc >= TARGETS['auc'])}). "
            f"KS {test_ks:.3f} (goal {TARGETS['ks']} or higher, {_verdict(test_ks >= TARGETS['ks'])}).")
    if psi is not None:
        text += f" Score PSI {psi:.4f} (goal below {TARGETS['psi']}, {_verdict(psi < TARGETS['psi'])})."
    if uplift:
        u = uplift['uplift']
        text += (f" Thin-filer AUC gain {u:+.3f}, 95% range {uplift['ci_low']:+.3f} to {uplift['ci_high']:+.3f} "
                 f"(goal +{TARGETS['thin_filer_uplift']} or more, {_verdict(u >= TARGETS['thin_filer_uplift'])}).")
    return text


# ---------- 결과 파일 → 쉬운 영어 요약 ----------

def performance_lines(m):
    out = [f"- AUC on test data: **{m['auc']:.3f}** (goal {TARGETS['auc']} or higher, {_verdict(m['auc'] >= TARGETS['auc'])}).",
           f"- KS on test data: **{m['ks']:.3f}** (goal {TARGETS['ks']} or higher, {_verdict(m['ks'] >= TARGETS['ks'])})."]
    if 'psi' in m:
        out.append(f"- Score PSI (train vs test): **{m['psi']:.4f}** (goal below {TARGETS['psi']}, {_verdict(m['psi'] < TARGETS['psi'])}).")
    fp = m.get('feature_psi')
    if fp:
        out.append(f"- Largest input PSI: {fp['max']:.4f} ({FEATURE_EN.get(fp['max_feature'], fp['max_feature'])}).")
    if all(k in m for k in ('precision', 'recall')):
        out.append(f"- Of people flagged as \"will default\", {m['precision']:.0%} really defaulted (precision). "
                   f"Of people who really defaulted, the model caught {m['recall']:.0%} (recall).")
    u = m.get('thin_filer_uplift')
    if u:
        out.append(f"- Thin-filers ({u['n_thin_filer']} people, {u['n_thin_filer_default']} defaults): AUC "
                   f"{u['auc_without_alt']:.3f} without alternative data → {u['auc_with_alt']:.3f} with it, "
                   f"gain **{u['uplift']:+.3f}** (95% range {u['ci_low']:+.3f} to {u['ci_high']:+.3f}).")
    return out


def _num(v):
    """보기 좋은 숫자: 정수면 정수, 아니면 소수 둘째 자리."""
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}"
    return str(v)


def explain_lines(e):
    reasons = e.get('rejection_reasons') or []
    if not reasons:
        return []
    out = ['Top reasons for one rejected customer (largest SHAP effect first):', '']
    for i, r in enumerate(reasons, 1):
        name = FEATURE_EN.get(r.get('feature'), r.get('feature'))
        line = f"{r.get('rank', i)}. {name}: {_num(r.get('current_value'))}"
        if r.get('reference_value') is not None:
            line += f" (same age group average: {_num(r['reference_value'])})"
        line += f", effect {r.get('shap_contribution', 0):+.2f}"
        out.append(line)
    return out


def _attr_line(name, a):
    rates = ', '.join(f"{g} {v:.1%}" for g, v in a.get('approval_rates', {}).items())
    return (f"- {name}: approval rates {rates}. DI ratio {a['di_ratio']:.3f} "
            f"(protected group: {a.get('protected_group')}).")


def fairness_lines(f, mitigation=None):
    ok_di = TARGETS['di_min'] <= f['di_ratio'] <= TARGETS['di_max']
    out = [f"- DI ratio: **{f['di_ratio']:.3f}** (fair range {TARGETS['di_min']} to {TARGETS['di_max']}, {_verdict(ok_di)}). "
           "DI = approval rate of the protected group divided by the approval rate of the others.",
           f"- Equalized odds gaps: TPR {f['equalized_odds_tpr_diff']:.3f}, FPR {f['equalized_odds_fpr_diff']:.3f} "
           f"(goal {TARGETS['eo_max']:.2f} or lower, "
           f"{_verdict(max(f['equalized_odds_tpr_diff'], f['equalized_odds_fpr_diff']) <= TARGETS['eo_max'])})."]
    labels = {'gender': 'Gender', 'age_group': 'Age group'}
    for key, a in (f.get('by_attribute') or {}).items():
        out.append(_attr_line(labels.get(key, key), a))
    if mitigation:
        b, a = mitigation['before'], mitigation['after']
        method = mitigation.get('method', '').replace('_', ' ')
        out.append(f"- Bias fix ({method}): DI {b['di_ratio']:.3f} → {a['di_ratio']:.3f}, "
                   f"TPR gap {b['equalized_odds_tpr_diff']:.3f} → {a['equalized_odds_tpr_diff']:.3f}, "
                   f"FPR gap {b['equalized_odds_fpr_diff']:.3f} → {a['equalized_odds_fpr_diff']:.3f}.")
        if mitigation.get('threshold_fit_on') == 'validation':
            out.append("  Group thresholds were chosen on validation data and only applied to test data.")
    return out


def _en_group(name):
    s = str(name)
    for ko in sorted(GROUP_EN, key=len, reverse=True):
        s = s.replace(ko, GROUP_EN[ko])
    return s


def _eta_words(eta):
    if eta >= 0.14:
        return 'large'
    if eta >= 0.06:
        return 'medium'
    if eta >= 0.01:
        return 'small'
    return 'very small'


def anova_lines(report):
    titles = {'test_1': 'Test 1 - do the input groups (financial / alternative / both) differ?',
              'test_2': 'Test 2 - do the model types (LR / XGBoost / LightGBM) differ?'}
    out = []
    for key in ('test_1', 'test_2'):
        t = report.get(key)
        if not t:
            continue
        means = ', '.join(f"{_en_group(g)} {v:.3f}" for g, v in t['group_means'].items())
        sig = t['p_value'] < TARGETS['alpha']
        out += [f"**{titles[key]}**", '',
                f"- Average AUC per group: {means}.",
                f"- F = {t['f_statistic']:.2f}, p = {t['p_value']:.2g}, eta squared = {t['effect_size_eta_squared']:.3f} "
                f"({_eta_words(t['effect_size_eta_squared'])} effect).",
                "- " + ("The groups are really different (p below 0.05)." if sig
                        else "No clear difference between the groups (p is 0.05 or higher).")]
        for c in t.get('post_hoc_tukey') or []:
            out.append(f"  - {_en_group(c['comparison'])}: difference {c['mean_diff']:+.3f}, adjusted p {c['p_adj']:.2g} "
                       f"({'different' if c['significant'] else 'not clearly different'}).")
        out.append('')
    return out


def summary_md(perf=None, explain=None, fairness=None, mitigation=None, anova=None):
    """결과 파일들을 쉬운 영어 요약 한 장으로."""
    lines = ['# Results in plain English', '']
    if perf:
        lines += ['## Model performance', ''] + performance_lines(perf) + ['']
    if explain:
        lines += ['## Why customers are rejected (SHAP)', ''] + explain_lines(explain) + ['']
    if fairness:
        lines += ['## Fairness', ''] + fairness_lines(fairness, mitigation) + ['']
    if anova:
        lines += ['## Statistical tests (one-way ANOVA)', ''] + anova_lines(anova)
    lines += ['', f'> {DISCLAIMER_EN}', '']
    return '\n'.join(lines)
