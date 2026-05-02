# require.md

# SDR + PyTorch 기반 딥러닝 변조 분류 및 데이터 복원 시스템 요구사항

## 1. 프로젝트 개요

본 프로젝트는 SDR 장비를 사용하여 BASK, BFSK, BPSK 이진 변조 신호를 송수신하고, 수신한 IQ 데이터를 기반으로 PyTorch 딥러닝 모델이 변조 방식을 자동 분류하는 시스템을 제작한다.

분류된 변조 방식에 따라 DSP 기반 복조기를 선택하여 송신 데이터를 복원하고, 잡음 및 간섭 조건에서 변조 분류 정확도와 데이터 수신 성능을 평가한다.

핵심 방향은 다음과 같다.

```text
딥러닝: 변조 방식 분류
DSP: 비트/문자 데이터 복원
```

처음부터 딥러닝으로 복조까지 수행하지 않는다. 1차 목표는 PyTorch 기반 1D CNN으로 BASK/BFSK/BPSK를 분류하고, 분류 결과에 따라 기존 DSP 복조기를 선택하는 구조이다.

---

## 2. 사용 하드웨어

### 2.1 송신 장비

- HackRF One
- 역할: BASK, BFSK, BPSK 변조 신호 송신
- 기본 송신 방식: Python 또는 GNU Radio 기반 신호 생성 후 HackRF로 송신

### 2.2 수신 장비

- RTL-SDR Blog V4
- 역할: 송신 신호 수신 및 IQ 샘플 저장
- 기본 수신 방식: Python 또는 GNU Radio 기반 IQ 캡처

### 2.3 연결 방식

권장 실험 방식은 다음과 같다.

```text
HackRF One TX → 감쇠기 attenuator → 동축 케이블 → RTL-SDR V4 RX
```

무선 안테나 송신은 전파 규제 문제가 발생할 수 있으므로, 초기 실험은 가능한 한 동축 케이블과 감쇠기를 사용한다.

---

## 3. 기본 송수신 파라미터

초기 구현 기준값은 아래와 같이 둔다. 추후 실험 조건에 따라 config 파일에서 변경 가능해야 한다.

| 항목 | 기본값 | 설명 |
|---|---:|---|
| Center Frequency | 433 MHz 또는 915 MHz | 실험 환경에 따라 선택 |
| TX Device | HackRF One | 송신기 |
| RX Device | RTL-SDR V4 | 수신기 |
| RTL-SDR Sample Rate | 2.4 MS/s | 수신 샘플레이트 |
| HackRF Sample Rate | 8 MS/s 권장 | 송신 샘플레이트 |
| Symbol Rate | 1 ksps ~ 10 ksps | 초기값 5 ksps 권장 |
| Samples per Symbol | sample_rate / symbol_rate | 예: 2.4e6 / 5e3 = 480 |
| Modulation | BASK, BFSK, BPSK | 이진 변조 |
| Payload | A, F, P 또는 랜덤 문자 | 실험 단계별 변경 |
| IQ Window Size | 1024 또는 2048 samples | 딥러닝 입력 단위 |
| IQ Format | complex64 또는 float32 I/Q 2채널 | PyTorch 입력용 |

---

## 4. 송신 데이터 및 프레임 구조

문자 하나만 단독 송신하지 말고, 동기화와 오류 검출을 위한 간단한 프레임 구조를 사용한다.

### 4.1 기본 프레임 구조

```text
[Preamble][Sync Word][Payload][CRC]
```

### 4.2 필드 정의

| 필드 | 예시 | 목적 |
|---|---|---|
| Preamble | 1010101010101010 | 타이밍 동기화 및 신호 검출 |
| Sync Word | 11001100 | 프레임 시작 위치 검출 |
| Payload | ASCII 문자 | 실제 전송 데이터 |
| CRC | CRC-8 또는 parity | 오류 검출 |

### 4.3 초기 문자 매핑

초기 검증 단계에서는 아래와 같이 단순 매핑을 사용한다.

| 변조 방식 | 송신 문자 | ASCII | bit sequence |
|---|---|---:|---|
| BASK | A | 0x41 | 01000001 |
| BFSK | F | 0x46 | 01000110 |
| BPSK | P | 0x50 | 01010000 |

단, 최종 실험에서는 변조 방식과 문자를 고정하지 않고 모든 변조 방식에서 다양한 payload를 보낼 수 있도록 구현한다.

좋은 최종 구조는 다음과 같다.

```text
BASK → A/F/P/random payload 송신 가능
BFSK → A/F/P/random payload 송신 가능
BPSK → A/F/P/random payload 송신 가능
```

이렇게 해야 딥러닝 모델이 특정 문자 파형을 외우는 것이 아니라 변조 방식의 특징을 학습하게 된다.

---

## 5. 변조 방식 정의

### 5.1 BASK

- Binary Amplitude Shift Keying
- bit 0과 bit 1을 서로 다른 진폭으로 표현한다.
- 예시:
  - bit 0: 낮은 진폭 또는 0
  - bit 1: 높은 진폭

### 5.2 BFSK

- Binary Frequency Shift Keying
- bit 0과 bit 1을 서로 다른 주파수 편이로 표현한다.
- 예시:
  - bit 0: carrier - delta_f
  - bit 1: carrier + delta_f

### 5.3 BPSK

- Binary Phase Shift Keying
- bit 0과 bit 1을 180도 위상 차이로 표현한다.
- 예시:
  - bit 0: phase 0
  - bit 1: phase pi

---

## 6. 딥러닝 입력 데이터

### 6.1 기본 입력 형태

1차 모델은 raw IQ 데이터를 사용한다.

```text
Input: raw IQ samples
Shape: [batch_size, 2, window_size]
Example: [batch_size, 2, 1024]
```

- channel 0: I component
- channel 1: Q component

### 6.2 저장 형식

권장 저장 형식은 `.npz`이다.

예시:

```python
np.savez(
    "sample_000001.npz",
    iq=iq_array,              # shape: [2, N]
    modulation="BPSK",
    payload="P",
    bits=bit_array,
    snr_db=10,
    sample_rate=2400000,
    symbol_rate=5000,
    center_freq=433000000
)
```

### 6.3 데이터 정규화

모델 입력 전 다음 정규화를 적용한다.

```python
iq = iq - iq.mean()
iq = iq / (np.std(iq) + 1e-8)
```

I/Q 각각에 대해 독립적으로 정규화하거나, complex magnitude 기준으로 정규화하는 방법을 비교할 수 있다.

### 6.4 Spectrogram 입력

1차 구현은 raw IQ 1D CNN으로 진행한다.

2차 확장 실험에서 spectrogram 기반 2D CNN을 추가할 수 있다.

```text
1차: raw IQ + 1D CNN
2차: spectrogram + 2D CNN
```

---

## 7. PyTorch 모델 요구사항

### 7.1 1차 모델: 1D CNN

기본 모델은 PyTorch 기반 1D CNN이다.

입력:

```text
[batch_size, 2, 1024]
```

출력:

```text
[batch_size, 3]
```

클래스:

```python
classes = ["BASK", "BFSK", "BPSK"]
```

### 7.2 권장 모델 구조

```text
Conv1d(2 → 32)
BatchNorm1d
ReLU
MaxPool1d

Conv1d(32 → 64)
BatchNorm1d
ReLU
MaxPool1d

Conv1d(64 → 128)
BatchNorm1d
ReLU

GlobalAveragePooling

Linear(128 → 64)
ReLU
Dropout

Linear(64 → 3)
```

### 7.3 Loss Function

```python
CrossEntropyLoss
```

### 7.4 Optimizer

```python
Adam 또는 AdamW
```

초기 권장값:

```python
learning_rate = 1e-3
batch_size = 64
epochs = 30
```

### 7.5 학습 결과 저장

모델 체크포인트는 다음 정보를 포함해야 한다.

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "class_names": ["BASK", "BFSK", "BPSK"],
    "window_size": 1024,
    "sample_rate": 2400000,
    "symbol_rate": 5000
}
```

---

## 8. 데이터셋 구성

### 8.1 데이터 종류

두 종류의 데이터셋을 만든다.

```text
1. Simulation Dataset
2. Real SDR Dataset
```

### 8.2 Simulation Dataset

Python으로 BASK/BFSK/BPSK baseband IQ를 생성한다.

포함 조건:

- clean signal
- AWGN added signal
- 다양한 SNR
- 다양한 payload
- 다양한 symbol rate
- 다양한 carrier frequency offset optional
- timing offset optional

### 8.3 Real SDR Dataset

실제 장비로 송수신하여 수집한다.

구조:

```text
HackRF One TX → attenuator/coaxial cable or short-range wireless → RTL-SDR V4 RX
```

저장 정보:

- IQ samples
- modulation label
- payload
- sample_rate
- symbol_rate
- center_freq
- tx_gain
- rx_gain
- distance 또는 attenuator 값
- capture time
- device info

### 8.4 데이터 분할

데이터는 단순 window 단위로 랜덤 분할하지 않는다. 같은 송신 세션에서 잘린 window가 train/test에 동시에 들어가면 데이터 누수가 발생할 수 있다.

권장 분할:

```text
Train: 70%
Validation: 15%
Test: 15%
```

분할 기준:

```text
session 단위 분할
```

예시:

```text
session_001 ~ session_070 → train
session_071 ~ session_085 → validation
session_086 ~ session_100 → test
```

---

## 9. AWGN 잡음 조건

SNR별 성능 평가를 수행한다.

### 9.1 SNR 범위

```text
SNR = -10, -5, 0, 5, 10, 15, 20 dB
```

### 9.2 AWGN 추가 방식

수신 IQ 또는 시뮬레이션 IQ에 대해 소프트웨어로 AWGN을 추가한다.

```python
def add_awgn(iq, snr_db):
    signal_power = np.mean(np.abs(iq) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*iq.shape) + 1j * np.random.randn(*iq.shape)
    )
    return iq + noise
```

### 9.3 평가 방식

각 SNR 조건마다 다음을 측정한다.

- Modulation classification accuracy
- Confusion matrix
- BER
- CER
- Packet success rate
- Throughput

---

## 10. 동시 신호 및 간섭 조건

“여러 신호를 동시에 송신”하는 실험은 난이도별로 나눈다.

### 10.1 1단계: 인접 채널 간섭

가장 현실적인 1차 간섭 실험이다.

```text
목표 신호: center_freq
간섭 신호: center_freq + delta_f
```

예시:

| 조건 | 목표 신호 | 간섭 신호 | 주파수 간격 |
|---|---|---|---:|
| Case 1 | BPSK | BFSK | 50 kHz |
| Case 2 | BPSK | BFSK | 100 kHz |
| Case 3 | BPSK | BFSK | 200 kHz |

측정 지표:

- SIR별 classification accuracy
- SIR별 BER
- 주파수 간격별 성능 변화

### 10.2 2단계: 다중 주파수 동시 송신

여러 변조 신호를 서로 다른 주파수에 배치한다.

```text
BASK: f_c - 200 kHz
BFSK: f_c
BPSK: f_c + 200 kHz
```

수신기는 넓은 대역을 수신한 뒤 각 채널을 필터링하여 분류한다.

### 10.3 3단계: 같은 주파수 중첩

동일 주파수에서 여러 변조가 동시에 섞이는 경우이다.

```text
BASK + BFSK + BPSK at same frequency
```

이 조건은 단순 변조 분류가 아니라 source separation 문제에 가까우므로 선택 실험으로 둔다.

필수 구현 범위에는 포함하지 않는다.

---

## 11. Baseline 비교

딥러닝 모델과 비교하기 위해 기존 DSP 특징 기반 분류기를 구현한다.

### 11.1 Feature-based Classifier

사용 가능한 특징:

| 특징 | 주로 유리한 변조 |
|---|---|
| 평균 에너지 / 진폭 분산 | BASK |
| 주파수 피크 개수 / 주파수 편이 | BFSK |
| 위상 변화 / phase transition | BPSK |

### 11.2 Baseline 구조

```text
IQ input
→ feature extraction
→ rule-based classifier 또는 classical ML
→ BASK/BFSK/BPSK output
```

### 11.3 비교 항목

| 항목 | Feature-based | 1D CNN |
|---|---|---|
| Clean accuracy | 측정 | 측정 |
| AWGN accuracy | 측정 | 측정 |
| Interference robustness | 측정 | 측정 |
| Inference latency | 측정 | 측정 |

---

## 12. 복조 및 데이터 복원

딥러닝 모델은 변조 방식만 분류한다. 실제 bit 복원은 분류 결과에 따라 DSP 복조기를 선택한다.

### 12.1 처리 흐름

```text
Received IQ
→ preprocessing
→ PyTorch 1D CNN modulation classifier
→ predicted modulation
→ select demodulator
→ bit recovery
→ frame sync
→ payload extraction
→ CRC check
→ recovered character
```

### 12.2 복조기

필요한 복조기:

```text
demod_bask()
demod_bfsk()
demod_bpsk()
```

각 함수는 다음을 반환한다.

```python
{
    "bits": recovered_bits,
    "payload": recovered_payload,
    "crc_ok": True or False,
    "ber": bit_error_rate
}
```

---

## 13. 성능 지표

### 13.1 변조 분류 성능

필수 지표:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- SNR별 Accuracy
- 간섭 조건별 Accuracy

### 13.2 데이터 복원 성능

필수 지표:

- BER: Bit Error Rate
- CER: Character Error Rate
- Packet Success Rate
- CRC Pass Rate

### 13.3 전송 성능

필수 지표:

- Throughput
- Effective throughput
- Symbol rate
- Payload success rate

Throughput 계산 예:

```text
throughput_bps = successfully_received_payload_bits / total_time_seconds
```

Effective throughput 계산 예:

```text
effective_throughput_bps = crc_pass_payload_bits / total_time_seconds
```

### 13.4 실시간 처리 성능

필수 지표:

- CNN inference latency
- windows per second
- CPU/GPU usage optional
- real-time possible 여부

---

## 14. 프로젝트 디렉터리 구조

권장 구조:

```text
sdr_modulation_ai/
│
├── require.md
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── sim/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── real/
│       ├── raw_iq/
│       ├── processed/
│       └── metadata/
│
├── src/
│   ├── config/
│   │   └── default_config.py
│   │
│   ├── signal/
│   │   ├── frame.py
│   │   ├── mod_bask.py
│   │   ├── mod_bfsk.py
│   │   ├── mod_bpsk.py
│   │   ├── demod_bask.py
│   │   ├── demod_bfsk.py
│   │   ├── demod_bpsk.py
│   │   └── awgn.py
│   │
│   ├── sdr/
│   │   ├── hackrf_tx.py
│   │   ├── rtlsdr_rx.py
│   │   └── capture_iq.py
│   │
│   ├── dataset/
│   │   ├── generate_sim_dataset.py
│   │   ├── collect_real_dataset.py
│   │   ├── iq_dataset.py
│   │   └── split_by_session.py
│   │
│   ├── models/
│   │   ├── cnn1d.py
│   │   └── spectrogram_cnn.py
│   │
│   ├── train/
│   │   ├── train_cnn1d.py
│   │   └── evaluate.py
│   │
│   ├── baseline/
│   │   ├── feature_extract.py
│   │   └── rule_classifier.py
│   │
│   └── app/
│       ├── realtime_receiver.py
│       └── realtime_inference.py
│
├── notebooks/
│   ├── 01_signal_visualization.ipynb
│   ├── 02_dataset_check.ipynb
│   └── 03_result_analysis.ipynb
│
├── results/
│   ├── checkpoints/
│   ├── figures/
│   ├── confusion_matrices/
│   └── logs/
│
└── scripts/
    ├── run_generate_sim.sh
    ├── run_train.sh
    ├── run_eval.sh
    └── run_capture_real.sh
```

---

## 15. Python 환경

### 15.1 권장 Python 버전

```text
Python 3.10 또는 3.11
```

### 15.2 PyTorch

CUDA 사용 가능 PC에서는 CUDA 지원 PyTorch를 사용한다.

예시:

```bash
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU 전용 환경에서는 다음을 사용한다.

```bash
python -m pip install torch torchvision torchaudio
```

### 15.3 requirements.txt 예시

```text
numpy
scipy
matplotlib
scikit-learn
pandas
pyyaml
tqdm
torch
torchvision
torchaudio
```

SDR 관련 패키지는 환경에 따라 별도 설치한다.

```text
pyrtlsdr
SoapySDR
hackrf
gnuradio optional
```

Windows 환경에서는 SDR 드라이버 설치와 Python 패키지 설치가 별도로 꼬일 수 있으므로, 처음에는 시뮬레이션 데이터셋부터 구현한다.

---

## 16. config.yaml 예시

```yaml
project:
  name: sdr_modulation_ai
  seed: 42

sdr:
  center_freq: 433000000
  tx_device: hackrf
  rx_device: rtlsdr
  tx_sample_rate: 8000000
  rx_sample_rate: 2400000
  symbol_rate: 5000
  tx_gain: 0
  rx_gain: 30

frame:
  preamble: "1010101010101010"
  sync_word: "11001100"
  crc: "crc8"

modulation:
  classes:
    - BASK
    - BFSK
    - BPSK
  payloads:
    - A
    - F
    - P

dataset:
  window_size: 1024
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  split_unit: session

awgn:
  snr_list: [-10, -5, 0, 5, 10, 15, 20]

model:
  type: cnn1d
  input_channels: 2
  num_classes: 3
  dropout: 0.3

train:
  batch_size: 64
  epochs: 30
  learning_rate: 0.001
  optimizer: adamw
  loss: cross_entropy
  device: cuda

evaluation:
  metrics:
    - accuracy
    - confusion_matrix
    - precision
    - recall
    - f1
    - ber
    - cer
    - throughput
```

---

## 17. 구현 순서

### Phase 1. 시뮬레이션 신호 생성

- BASK/BFSK/BPSK baseband IQ 생성
- frame 생성
- payload bit 변환
- AWGN 추가
- `.npz` 데이터셋 저장

완료 조건:

```text
각 변조 방식별 IQ 파형을 생성하고 시각화할 수 있어야 한다.
```

### Phase 2. PyTorch Dataset 구현

- `.npz` 파일 로더 구현
- IQ tensor shape `[2, N]` 반환
- label index 반환
- train/val/test 분리

완료 조건:

```text
DataLoader에서 batch shape [B, 2, 1024]가 정상 출력되어야 한다.
```

### Phase 3. 1D CNN 학습

- cnn1d.py 구현
- train_cnn1d.py 구현
- validation accuracy 출력
- checkpoint 저장

완료 조건:

```text
clean simulation dataset에서 95% 이상의 분류 정확도를 목표로 한다.
```

### Phase 4. AWGN 성능 평가

- SNR별 test set 생성
- SNR별 accuracy 측정
- confusion matrix 저장
- 결과 그래프 생성

완료 조건:

```text
SNR 변화에 따른 accuracy curve를 생성해야 한다.
```

### Phase 5. DSP 복조기 구현

- BASK 복조
- BFSK 복조
- BPSK 복조
- frame sync
- CRC check
- payload 복원

완료 조건:

```text
clean signal에서 A/F/P 문자가 정상 복원되어야 한다.
```

### Phase 6. 실제 SDR 데이터 수집

- HackRF 송신
- RTL-SDR 수신
- IQ 캡처
- metadata 저장

완료 조건:

```text
실제 수신 IQ에서 변조별 데이터셋을 수집할 수 있어야 한다.
```

### Phase 7. 실제 SDR 데이터 평가

- simulation으로 학습한 모델을 real SDR 데이터에 적용
- 필요 시 real data fine-tuning
- real 환경 accuracy, BER, CER 측정

완료 조건:

```text
실제 SDR 수신 데이터에서도 변조 분류 및 문자 복원이 가능해야 한다.
```

### Phase 8. 간섭 실험

- 인접 채널 간섭 추가
- delta_f별 성능 측정
- SIR별 성능 측정

완료 조건:

```text
간섭 조건에서 accuracy, BER 변화 그래프를 생성해야 한다.
```

---

## 18. 결과물

최종 결과물은 다음을 포함해야 한다.

```text
1. 시뮬레이션 IQ 데이터 생성 코드
2. 실제 SDR IQ 캡처 코드
3. PyTorch Dataset 코드
4. 1D CNN 모델 코드
5. 학습 코드
6. 평가 코드
7. DSP baseline 분류기
8. BASK/BFSK/BPSK 복조기
9. SNR별 성능 그래프
10. Confusion Matrix
11. BER/CER/Throughput 결과표
12. 실험 보고서 또는 논문 초안
```

---

## 19. 안전 및 전파 규제 조건

### 19.1 기본 원칙

- 허가되지 않은 주파수에서 장시간 송신하지 않는다.
- 출력 gain을 최소화한다.
- 초기 실험은 안테나 방사가 아니라 동축 케이블 기반으로 진행한다.
- HackRF TX와 RTL-SDR RX 직결 시 감쇠기를 사용한다.
- 수신기 과입력을 방지한다.

### 19.2 권장 연결

```text
HackRF TX → 30 dB ~ 60 dB attenuator → RTL-SDR RX
```

### 19.3 무선 송신 시 주의

- 가능한 ISM 대역을 사용한다.
- 송신 시간을 짧게 유지한다.
- 주변 통신 시스템에 간섭을 주지 않는다.
- 학교/연구실/국가 전파 규정을 따른다.

---

## 20. Claude Code / Codex 작업 지시사항

이 프로젝트를 구현할 때 다음 원칙을 따른다.

### 20.1 구현 우선순위

1. 시뮬레이션 데이터 생성
2. PyTorch Dataset
3. 1D CNN 학습
4. AWGN 평가
5. DSP 복조기
6. SDR 실제 수집
7. 실제 데이터 평가
8. 간섭 실험

### 20.2 코드 작성 원칙

- 기존 기능을 제거하지 않는다.
- 각 모듈은 독립 실행 가능하게 작성한다.
- config.yaml로 주요 파라미터를 관리한다.
- 하드코딩을 최소화한다.
- 실험 결과는 results/ 하위에 저장한다.
- 학습 로그와 평가 결과를 CSV 또는 JSON으로 저장한다.
- 모든 random seed를 고정할 수 있게 한다.

### 20.3 금지사항

- 처음부터 복잡한 Transformer 모델을 도입하지 않는다.
- 처음부터 같은 주파수 중첩 신호 분리를 필수 기능으로 만들지 않는다.
- 딥러닝으로 비트 복원까지 한 번에 처리하려고 하지 않는다.
- 실제 SDR 송신부터 시작하지 않는다.
- 시뮬레이션 검증 없이 하드웨어 실험으로 넘어가지 않는다.

### 20.4 우선 구현해야 할 파일

```text
src/signal/frame.py
src/signal/mod_bask.py
src/signal/mod_bfsk.py
src/signal/mod_bpsk.py
src/signal/awgn.py
src/dataset/generate_sim_dataset.py
src/dataset/iq_dataset.py
src/models/cnn1d.py
src/train/train_cnn1d.py
src/train/evaluate.py
```

---

## 21. 최종 목표 문장

본 프로젝트의 최종 목표는 다음과 같다.

```text
HackRF One으로 BASK/BFSK/BPSK 이진 변조 신호를 송신하고,
RTL-SDR V4로 수신한 IQ 데이터를 PyTorch 1D CNN 모델에 입력하여
변조 방식을 자동 분류한다.
이후 분류 결과에 따라 DSP 복조기를 선택하여 payload 문자를 복원하고,
AWGN 및 인접 채널 간섭 조건에서 Accuracy, Confusion Matrix, BER, CER,
Throughput을 측정하여 딥러닝 기반 변조 분류 시스템의 성능을 평가한다.
```
