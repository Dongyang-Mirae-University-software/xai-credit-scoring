"""시뮬레이터 테스트 스크립트"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.data.loader import load_gmsc
from src.data.simulator import generate_alternative_data

# 데이터 로드
print("=" * 80)
print("데이터 로드 중...")
print("=" * 80)
train_df, test_df = load_gmsc()

# 대안 데이터 생성
print("\n" + "=" * 80)
print("대안 데이터 생성 중...")
print("=" * 80)
train_df_alt = generate_alternative_data(train_df)

# 결과 확인
print("\n" + "=" * 80)
print("최종 결과")
print("=" * 80)
new_cols = ['telecom_payment_rate', 'utility_payment_rate',
            'spending_consistency', 'regular_payment_count',
            'app_login_frequency', 'is_thin_filer']
print(f"새로 추가된 컬럼: {new_cols}")
print(f"\n훈련 데이터 샘플:")
print(train_df_alt[['SeriousDlqin2yrs'] + new_cols[:5]].head())
print(f"\n데이터 형태: {train_df_alt.shape}")
print(f"전체 컬럼 수: {len(train_df_alt.columns)}")
