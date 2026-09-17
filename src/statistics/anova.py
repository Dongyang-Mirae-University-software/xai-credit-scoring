"""
ANOVA 통계 검증.
검증 1: 피처 그룹별 모델(전통/대안/통합) One-way ANOVA
검증 2: 알고리즘별(LR/XGB/LGBM) One-way ANOVA
p < 0.05일 경우 Tukey HSD 사후 검정을 포함한다.
"""


def run_anova_feature_group(*args, **kwargs):
    raise NotImplementedError


def run_anova_algorithm(*args, **kwargs):
    raise NotImplementedError
