"""순수 방법 1 테스트 - 공분산 행렬 기반 샘플링만 사용"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from src.data.loader import load_gmsc

# 데이터 로드
print("=" * 80)
print("데이터 로드 중...")
print("=" * 80)
train_df, test_df = load_gmsc()

# 순수 방법 1 테스트
print("\n" + "=" * 80)
print("순수 방법 1 테스트 (공분산 행렬 기반 샘플링만)")
print("=" * 80)

n = len(train_df)
target = train_df['SeriousDlqin2yrs'].values.astype(float)

# Target과의 상관계수 설정 (0.30~0.50)
target_correlations = {
    'telecom_payment_rate': np.random.uniform(0.30, 0.50),
    'utility_payment_rate': np.random.uniform(0.30, 0.50),
    'spending_consistency': np.random.uniform(0.30, 0.50),
    'regular_payment_count': np.random.uniform(0.30, 0.50),
    'app_login_frequency': np.random.uniform(0.30, 0.50)
}

print(f"\n설정된 목표 상관계수:")
for var, corr in target_correlations.items():
    print(f"   {var}: {corr:.4f}")

# 공분산 행렬 구성
cov_matrix = np.eye(6)

# 타겟과의 상관관계
for idx, corr in enumerate(target_correlations.values(), start=1):
    cov_matrix[0, idx] = corr
    cov_matrix[idx, 0] = corr

# 변수 간 상관관계
for i in range(1, 6):
    for j in range(i+1, 6):
        cov_matrix[i, j] = np.random.uniform(0.20, 0.40)
        cov_matrix[j, i] = cov_matrix[i, j]

# 다변량 정규분포에서 샘플링 (순수 방법 1: 이게 전부)
mean = np.array([0, 0.85, 0.80, 75, 8, 15])
std_devs = np.array([1, 0.15, 0.15, 12, 4, 6])
scaled_cov = np.diag(std_devs) @ cov_matrix @ np.diag(std_devs)

try:
    samples = np.random.multivariate_normal(mean, scaled_cov, n)
    print(f"\n공분산 행렬 기반 샘플링 완료")
except np.linalg.LinAlgError:
    print(f"공분산 행렬 오류, 표준 정규분포 사용")
    samples = np.random.standard_normal((n, 6)) * std_devs + mean

# 샘플에서 직접 추출 (추가 조정 없음)
train_df_alt = train_df.copy()
train_df_alt['telecom_payment_rate'] = np.clip(samples[:, 1], 0, 1)
train_df_alt['utility_payment_rate'] = np.clip(samples[:, 2], 0, 1)
train_df_alt['spending_consistency'] = np.clip(samples[:, 3], 0, 100)
train_df_alt['regular_payment_count'] = np.maximum(0, np.round(samples[:, 4]).astype(int))
train_df_alt['app_login_frequency'] = np.maximum(0, np.round(samples[:, 5]).astype(int))

# 실제 상관계수 계산
alt_vars = ['telecom_payment_rate', 'utility_payment_rate',
            'spending_consistency', 'regular_payment_count',
            'app_login_frequency']

print(f"\n{'='*75}")
print(f"생성된 대안 변수와 실제 Target(SeriousDlqin2yrs) 간 실제 상관계수")
print(f"{'='*75}")
print(f"{'변수명':<30} {'실제 상관계수':<18} {'범위 확인':<20}")
print("-" * 75)

correlations = {}
in_range_count = 0

for var in alt_vars:
    corr = train_df_alt[var].corr(train_df_alt['SeriousDlqin2yrs'])
    correlations[var] = corr

    abs_corr = abs(corr)
    if 0.30 <= abs_corr <= 0.50:
        status = "OK (0.30~0.50)"
        in_range_count += 1
    else:
        status = f"OUT ({abs_corr:.4f})"

    print(f"{var:<30} {corr:>14.4f}    {status:<20}")

print("-" * 75)
avg_abs_corr = np.mean([abs(c) for c in correlations.values()])
print(f"{'평균 상관계수 절대값':<30} {avg_abs_corr:>14.4f}")
print(f"{'범위 내 변수 개수':<30} {in_range_count}/5")
print(f"{'='*75}\n")

# 결론
if in_range_count == 5:
    print("✅ 순수 방법 1로 상관계수 0.3~0.5 범위 달성!")
else:
    print(f"❌ 순수 방법 1에서는 {in_range_count}/5개만 범위 내")
    print(f"   평균: {avg_abs_corr:.4f} (목표: 0.3~0.5)")
