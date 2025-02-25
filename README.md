# 에너지 시스템 통합 리포지토리

## 개요
이 리포지토리는 다양한 보일러 및 열 펌프 모델을 구현하여 에너지 시스템 통합을 다룹니다. 각 모델은 에너지, 엔트로피, 엑서지 균형을 분석하는 기능을 포함하고 있습니다.

## 파일 설명

- `constant.py`: 온도, 길이, 에너지, 전력 변환을 위한 다양한 단위 변환 상수 및 함수 정의.
- `En_system_intergrated.py`: 전기 보일러, 가스 보일러, 열 펌프 등 다양한 난방 시스템 모델을 구현하며, 열역학 분석과 균형 계산을 수행.

## 설치 방법
모델을 사용하려면 필요한 패키지를 설치해야 합니다. 다음 명령어를 실행하세요:

```bash
uv add git+https://github.com/BET-lab/En_system_Ex_analysis.git

uv sync
```

## 사용 방법

### 모듈 임포트하기
```python
import numpy as np
import math
import constant as c
from En_system_intergrated import ElectricBoiler, GasBoiler, HeatPumpBoiler
```

### 예제: 전기 보일러 시스템 생성
```python
boiler = ElectricBoiler()
boiler.system_update()
print(boiler.energy_balance)
```

### 예제: 총 엑서지 사용량 계산
```python
from En_system_intergrated import calculate_total_exergy_consumption

total_exergy = calculate_total_exergy_consumption(boiler.exergy_balance)
print(f"총 엑서지 사용량: {total_exergy} W")
```
