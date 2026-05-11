# 프로젝트 필수 규칙

이 문서는 SDR + 딥러닝 변조 분류 및 데이터 복원 프로젝트에서 반드시 지켜야 하는 규칙만 정리한다.

## 1. 목표

```text
SDR로 BASK/BFSK/BPSK 신호를 송수신한다.
수신 IQ를 딥러닝 모델에 넣어 변조 방식을 분류한다.
분류 결과에 따라 DSP 복조기를 선택한다.
payload, CRC, BER, CER, Packet Success Rate를 평가한다.
```

딥러닝은 우선 변조 분류를 담당한다. 비트/문자 복원은 DSP 복조기와 frame parser가 담당한다.

## 2. 장비와 RF 조건 기록

기본 장비:

```text
TX: HackRF One
RX: RTL-SDR Blog V4
OS: Windows
SDR 환경: Radioconda + SoapySDR
Python: 프로젝트 .venv Python 3.11
```

실험 1~3은 모두 RF 케이블 연결이 아니라 OTA 안테나 송수신이었다.

```text
Exp1: 약 10 cm OTA, side-by-side
Exp2: 약 10 cm OTA, side-by-side
Exp3: 약 1 m OTA, face-to-face
Exp4: 약 1 m OTA, face-to-face 예정
```

유선 RF 연결 실험을 하면 반드시 별도 실험으로 분리하고 metadata에 `rf_path`, `rf_cable_between_sdr`, attenuator 값을 명시한다.

## 3. 디렉터리 규칙

루트에는 공통 문서만 둔다.

```text
SDR/
  README.md
  require.md
  requirements.txt
  config.yaml
  experiments/
  reference/
```

루트에는 다음 폴더를 다시 만들지 않는다.

```text
src/
tests/
data/
results/
```

모든 새 실험은 새 폴더에서 독립적으로 진행한다.

```text
experiments/expN/
  README.md
  PLAN.md
  FINAL_REPORT.md
  config/
  sourcecode/
  tests/
  data/
  results/
```

## 4. 데이터 규칙

payload는 modulation별로 고정하지 않는다. 모든 modulation에서 같은 payload pool을 공유한다.

```text
["A", "F", "P", "0", "1", "7", "K", "R", "S", "Z"]
```

frame 기본 구조:

```text
preamble 16 bits: 1010101010101010
sync word 8 bits: 11001100
payload 8 bits
CRC8 8 bits
total 40 bits
```

## 5. Split 규칙

최종 성능 판단은 random-window split이 아니라 session-held-out split을 우선한다.

금지:

```text
같은 raw capture 또는 같은 session에서 나온 window가 train/test에 동시에 들어가는 것
```

random-window split은 smoke test 또는 baseline 확인에만 사용한다.

## 6. 모델 입력 규칙

기본 입력:

```text
[batch, 2, 1024]
channels = [I, Q]
```

실험 3 이후 권장 입력:

```text
[batch, 3, 1024]
channels = [I, Q, instantaneous_frequency]
```

class 순서는 항상 고정한다.

```text
["BASK", "BFSK", "BPSK"]
```

## 7. 평가 규칙

변조 분류 평가는 다음을 포함한다.

```text
Accuracy
Confusion Matrix
Precision
Recall
F1
class별 recall
session/payload/gain/offset/SNR 조건별 accuracy
```

실험 4부터 데이터 복원 평가는 다음을 포함한다.

```text
BER
CER
CRC Pass Rate
Packet Success Rate
Payload Recovery Accuracy
Oracle-demod Recovery Rate
```

## 8. Metadata 규칙

실제 SDR capture에는 최소 다음 정보를 저장한다.

```text
session_id
rf_path
rf_cable_between_sdr
tx_rx_distance_m
antenna_layout
center_freq
baseband_offset_hz
tx_sample_rate
rx_sample_rate
symbol_rate
tx_gain / tx_vga_gain / tx_amp_gain
rx_gain
modulation
payload
estimated_snr_db 가능하면 기록
```

OTA 실험에서는 가능하면 다음도 기록한다.

```text
antenna orientation
line_of_sight
near_metal_objects
human_nearby
```

## 9. Git 규칙

GitHub에는 대용량 산출물을 올리지 않는다.

```text
*.bin
*.npz
*.pt
*.pth
raw IQ
processed dataset
large checkpoint
```

공유가 필요하면 파일 위치와 재생성 명령만 문서에 남긴다.

## 10. 문서화 규칙

새 실험이 끝나면 반드시 작성한다.

```text
experiments/expN/FINAL_REPORT.md
SDR/README.md 실험 요약 표 업데이트
```

보고서에는 목적, RF 조건, 데이터 생성 방식, split 방식, 실행 명령, 시행착오, 결과, 다음 과제를 포함한다.
