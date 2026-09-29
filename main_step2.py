"""
Step 2: 모델 3중 묶음 학습
이재희 담당 작업의 두 번째 단계를 실행하는 스크립트
"""

import logging
from src.models.train import main

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    main()
