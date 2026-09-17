"""
GMSC(Give Me Some Credit) 스키마를 따르는 목(mock) 데이터 생성 스크립트.
실제 Kaggle 다운로드 전, 로더·전처리·시뮬레이터 개발/테스트용.
실행: python scripts/generate_mock_data.py
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N = 500
OUT_PATH = "data/mock/cs-training-mock.csv"


def generate(n=N, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    # 정상 93.3% / 연체 6.7% 비율(원본 GMSC 실제 분포)에 맞춰 생성
    target = rng.choice([0, 1], size=n, p=[0.933, 0.067])

    age = rng.integers(21, 80, size=n)
    # 연체자는 평균적으로 연체 이력·부채비율이 더 높게 나오도록 target과 약하게 연동
    revolving_util = np.clip(rng.normal(0.3 + 0.2 * target, 0.25, n), 0, 1.5)
    n_30_59 = rng.poisson(0.3 + 1.5 * target, n)
    n_60_89 = rng.poisson(0.1 + 0.8 * target, n)
    n_90 = rng.poisson(0.1 + 1.0 * target, n)
    debt_ratio = np.clip(rng.normal(0.3 + 0.15 * target, 0.2, n), 0, 5)
    monthly_income = np.clip(rng.normal(4500 - 800 * target, 2000, n), 0, None)
    open_credit_lines = rng.integers(0, 20, size=n)
    real_estate_loans = rng.integers(0, 5, size=n)
    dependents = rng.integers(0, 5, size=n)

    df = pd.DataFrame({
        "SeriousDlqin2yrs": target,
        "RevolvingUtilizationOfUnsecuredLines": revolving_util.round(4),
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": n_30_59,
        "DebtRatio": debt_ratio.round(4),
        "MonthlyIncome": monthly_income.round(2),
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": n_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": n_60_89,
        "NumberOfDependents": dependents.astype(float),
    })

    # 실제 GMSC처럼 일부 컬럼에 결측치 주입 (MonthlyIncome ~20%, NumberOfDependents ~2.6%)
    income_missing_idx = rng.choice(n, size=int(n * 0.198), replace=False)
    df.loc[income_missing_idx, "MonthlyIncome"] = np.nan

    dep_missing_idx = rng.choice(n, size=int(n * 0.026), replace=False)
    df.loc[dep_missing_idx, "NumberOfDependents"] = np.nan

    df.index.name = ""  # 원본 GMSC처럼 첫 컬럼이 무명 행 번호(row_id)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH)
    print(f"생성 완료: {OUT_PATH} ({len(df)}행 x {len(df.columns)}컬럼)")
    print(df["SeriousDlqin2yrs"].value_counts(normalize=True))
