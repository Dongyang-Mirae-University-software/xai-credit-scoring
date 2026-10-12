"""방법 2, 3 테스트"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.data.loader import load_gmsc
from src.data.simulator_2 import generate_alternative_data_method2
from src.data.simulator_3 import generate_alternative_data_method3

print("=" * 80)
print("데이터 로드 중...")
print("=" * 80)
train_df, test_df = load_gmsc()

# 방법 2 테스트
print("\n" + "=" * 80)
print("방법 2 테스트 (Cholesky 분해)")
print("=" * 80)
train_df_m2 = generate_alternative_data_method2(train_df)

# 방법 3 테스트
print("\n" + "=" * 80)
print("방법 3 테스트 (Gaussian Copula)")
print("=" * 80)
train_df_m3 = generate_alternative_data_method3(train_df)

print("\n" + "=" * 80)
print("테스트 완료")
print("=" * 80)
