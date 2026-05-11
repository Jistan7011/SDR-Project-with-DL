# 실험 2 통합 계획

작성일: 2026-05-05

이 문서는 실험 2에서 실제로 따른 단일 실험 계획입니다. 이전의 초안 계획과 개선 계획은 이 문서로 통합했습니다. 최종 결과와 분석은 `FINAL_REPORT_EXP02.md`에 정리되어 있습니다.

## 1. 목적

실험 2의 목적은 실제 SDR capture에서 얻은 IQ 데이터로 학습한 BASK/BFSK/BPSK 변조 분류 모델이 보지 못한 capture session에서도 일반화되는지 검증하는 것입니다.

실험 1은 같은 real capture 파일에서 random window를 뽑아 train/test를 구성했습니다. 이 방식은 baseline으로는 유용했지만, 같은 RF 조건의 window가 train과 test에 동시에 들어갈 수 있었습니다. 따라서 실험 2에서는 더 엄격한 `session-held-out` 평가를 사용합니다.

## 2. 핵심 가설

모델이 진짜 변조 특징을 학습했다면, 학습에 사용하지 않은 새로운 session에서도 BASK/BFSK/BPSK를 분류할 수 있어야 합니다.

반대로 session-held-out 조건에서 성능이 무너지면, 모델이 변조 방식이 아니라 gain, offset, payload 패턴, 간섭, 특정 channel condition 같은 capture-specific artifact를 학습했을 가능성이 큽니다.

## 3. Session 정의

session은 독립적인 RF capture 조건 묶음입니다. 같은 raw capture에서 window만 여러 개 뽑는 것은 새로운 session이 아닙니다.

각 session은 아래 조건 중 하나 이상이 달라져야 합니다.

- capture 시간
- TX/RX gain
- baseband offset
- 안테나 위치 또는 방향 재배치
- 장비 재시작
- TX/RX 위치 또는 배치

최종 split:

```text
train: session_001 ~ session_009
val:   session_010 ~ session_012
test:  session_013 ~ session_015
```

어떤 session도 둘 이상의 split에 동시에 들어가면 안 됩니다.

## 4. 장비 및 신호 조건

```text
TX: HackRF One
RX: RTL-SDR Blog V4
center frequency: 433 MHz
TX sample rate: 8 MS/s
RX sample rate: 2.4 MS/s
symbol rate: 5 ksps
processed target sample rate: 160 kS/s
window size: 1024 IQ samples
rf_path: OTA antenna
tx_rx_distance: about 10 cm
antenna_layout: side_by_side
```

실험 2도 실험 1과 동일하게 HackRF One과 RTL-SDR V4를 유선 RF 케이블로 연결하지 않았다.
두 장비 모두 안테나를 사용했고, 약 `10 cm` 거리에서 옆에 나란히 세운 OTA 근거리
배치로 session을 수집했다. 따라서 실험 2의 session-held-out 결과는 `10 cm OTA 조건`
내에서의 session 일반화 성능이다.

payload pool:

```text
["A", "F", "P", "0", "1", "7", "K", "R", "S", "Z"]
```

모든 modulation class가 같은 payload pool을 사용합니다. 이렇게 해야 모델이 “payload A면 BASK” 같은 shortcut을 외우는 문제를 막을 수 있습니다.

frame 구조:

```text
preamble:  1010101010101010   16 bits
sync word: 11001100           8 bits
payload:   ASCII 문자 1개      8 bits
CRC8:      8 bits
total:     40 bits
```

## 5. 데이터 파이프라인

전체 흐름:

```text
HackRF TX
-> RF path
-> RTL-SDR RX raw complex64 IQ
-> baseband offset 기준 channelize
-> low-pass filter
-> 160 kS/s로 downsample
-> 1024-sample window 추출
-> [I, Q] 입력으로 저장
```

모델 입력 shape:

```text
[batch, 2, 1024]
```

정답 label은 capture 명령에서 지정한 modulation class와 매칭합니다.

```text
BASK -> class 0
BFSK -> class 1
BPSK -> class 2
```

## 6. 실제 반영된 계획 수정 사항

실험 2 진행 중 다음 수정이 실제로 반영되었습니다.

- payload memorization을 막기 위해 모든 modulation에서 payload pool을 공유했습니다.
- random-window split 대신 session-held-out split을 사용했습니다.
- test session을 3개 확보하기 위해 session 수를 15개로 늘렸습니다.
- session quality와 estimated SNR 확인을 위해 TX-off noise-only capture를 추가했습니다.
- DC spike 위험 때문에 100 kHz offset은 primary 조건에서 제외했습니다.
- primary offset 후보는 250 kHz와 500 kHz로 두었습니다.
- RTL-SDR 2.4 MS/s 안정성을 capture gate로 확인했습니다.
- 모델 전환 기준을 미리 정하고 CNN1D, augmentation, VGG1D, ResNet1D 순서로 비교했습니다.

## 7. 모델 전환 기준

시작 모델은 CNN1D입니다.

아래 조건 중 하나라도 해당하면 다음 모델로 확장합니다.

```text
mean val accuracy < 0.70
또는 class별 recall 중 하나라도 < 0.55
```

전환 순서:

```text
CNN1D
-> CNN1D + augmentation
-> VGG-style 1D CNN
-> ResNet1D
-> ResNet1D + instantaneous-frequency feature ablation
```

이 이후의 feature fusion은 실험 2 범위가 아니라 실험 3으로 넘깁니다.

## 8. 실행 순서

모든 명령은 아래 위치에서 실행합니다.

```powershell
cd D:\ai_projects\SDR\experiments\exp2\sourcecode
```

테스트:

```powershell
..\..\..\.venv\Scripts\python.exe -m pytest ..\tests\root_tests_snapshot -q
```

15-session dataset import:

```powershell
if (Test-Path ..\data\processed) { Remove-Item -Recurse -Force ..\data\processed }

..\..\..\.venv\Scripts\python.exe -m src.dataset.import_exp02_sessions --min-sessions 15
```

공식 모델 학습:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp02.yaml --model-type resnet1d --output-root ..\results\exp02_15session_resnet
```

오류 분석:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_resnet_errors --config ..\config\config.exp02.yaml --data-root ..\data\processed --output-dir ..\results\reports
```

최종 그림 재생성:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.make_final_report_figures
```

## 9. 성공 기준

목표:

```text
session-held-out test accuracy >= 0.75
class recall >= 0.65
complete class collapse가 없을 것
```

최종 선택 모델:

```text
ResNet1D [I,Q]
```

선택 이유:

- session-held-out accuracy 기준을 넘었습니다.
- 비교한 모델 중 공식 기준으로 가장 적절한 구조였습니다.
- instantaneous-frequency ablation은 worst-class recall을 개선했지만 overall accuracy를 낮췄기 때문에 공식 모델이 아니라 분석 결과로만 남겼습니다.

## 10. 최종 상태

실험 2는 완료 상태입니다.

최종 결과:

```text
official model: ResNet1D [I,Q]
mean accuracy: 0.7579
worst observed recall: 0.6558
ensemble accuracy: 0.7594
```

남은 문제:

```text
보지 못한 session에서 BPSK와 BFSK가 BASK로 혼동되는 경우가 아직 남아 있습니다.
```

다음 단계:

```text
실험 3에서는 OTA 거리를 약 `1 m`로 늘린 신규 session을 수집하고, feature fusion이
`10 cm -> 1 m` 거리 변화에서 성능 저하를 줄이는지 검토합니다.
```
