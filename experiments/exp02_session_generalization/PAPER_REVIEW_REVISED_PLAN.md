# 논문 기반 현재 실험 검토 및 실험 2 수정 계획

작성 기준일: 2026-05-04

## 1. 검토한 논문과 핵심 메시지

`paper/` 폴더의 AMC 및 RadioML 계열 논문을 현재 프로젝트 기준으로 검토했다. `WRONG_PAPER_수학논문_Rayner...pdf`는 SDR/AMC와 무관하므로 제외했다.

### 1.1 O'Shea 2016, Convolutional Radio Modulation Recognition Networks

핵심:

- AMC 입력은 complex baseband time series이며, I/Q를 2개 real channel로 보고 CNN에 넣는다.
- RadioML류 데이터셋은 짧은 time-window sample과 modulation/SNR label로 구성한다.
- 단순 AWGN만이 아니라 carrier frequency offset, sample clock drift, multipath fading, random scale/translation/phase/noise를 포함한 channel model을 사용한다.
- 학습 목표는 payload 복원이 아니라 "신호가 어떤 modulation으로 생성되었는지" 분류하는 것이다.
- 논문은 128 sample window를 많이 사용하고, 이후 OTA 논문에서는 1024 sample example도 사용한다.

현재 실험과의 관계:

- 현재 프로젝트의 `[batch, 2, 1024]` 1D CNN 입력 설계는 논문 방향과 일치한다.
- 그러나 현재 simulation channel은 AWGN 중심이라 논문 수준의 channel realism이 부족하다.
- 현재 real 실험은 같은 capture에서 random window를 train/val/test로 나누므로 일반화 검증으로는 약하다.

### 1.2 O'Shea 2018, Over-the-Air Deep Learning Based Radio Signal Classification

핵심:

- simulation impairment와 over-the-air measurement를 분리해 비교한다.
- OTA 실험에서는 SDR 2대를 사용해 실제 송수신하고, 수신 sample을 ground truth modulation label과 함께 저장한다.
- direct conversion DC impairment를 피하기 위해 신호를 약 1 MHz off-tune해서 수신하는 설계를 사용한다.
- CNN/ResNet은 raw I/Q sample을 unit variance로 normalize한 뒤 expert feature 없이 학습한다.
- 1024 sample example을 사용하며, 1D VGG식 CNN/ResNet이 baseline feature method보다 좋은 성능을 낸다.
- 실험은 SNR, CFO, SRO, fading 같은 impairment별 성능 곡선을 비교한다.

현재 실험과의 관계:

- 실제 HackRF/RTL-SDR capture를 사용한 것은 논문 방향과 맞다.
- 하지만 현재는 `BASK=A`, `BFSK=F`, `BPSK=P`로 payload가 class와 묶여 있어 class leakage 위험이 있다.
- 현재는 session-level split이 아니라 window-level random split이므로 OTA 일반화 성능으로 보기 어렵다.
- 현재 100 kHz offset은 DC를 피하려는 의도가 있으나, 논문식으로는 offset 후보를 sweep하고 SNR/CFO/fading 조건을 metadata로 관리해야 한다.

### 1.3 O'Shea 2016, Radio Transformer Networks

핵심:

- 실제 무선 수신에서는 time offset, frequency offset, phase offset, sample timing offset이 존재한다.
- 동기화는 일종의 attention 문제이며, RTN은 classifier 앞에서 신호를 canonical form으로 보정하도록 학습한다.

현재 실험과의 관계:

- 현재 전처리는 평균 제거/표준편차 정규화 정도다.
- real capture에서 주파수 peak가 흔들리거나 DC/offset 문제가 생길 수 있으므로, 실험 2에는 explicit channelization 또는 learnable/algorithmic synchronization이 필요하다.

### 1.4 Zhou 2019, Robust Modulation Classification Using CNN

핵심:

- received signal을 직접 CNN에 넣어 robust feature를 학습한다.
- SNR별 학습/평가, confusion matrix, 중간 feature map 분석이 중요하다.
- amplitude 계열 modulation은 낮은 SNR에서 더 취약하다.
- VGG식 deeper CNN, BatchNorm, pooling, global average pooling 등을 사용한다.

현재 실험과의 관계:

- 현재 CNN은 얕은 1D CNN으로 baseline에는 충분하지만, 논문 비교용으로는 VGG/ResNet 계열 확장이 필요하다.
- BASK는 amplitude 기반이라 실제 noise/gain 변화에 취약할 수 있으므로 SNR/gain별 분석이 필요하다.

### 1.5 Han 2021, Deep Feature Fusion for High Noise and Large Dynamic Input

핵심:

- high noise, large dynamic input 조건에서는 단일 raw IQ feature만으로 부족할 수 있다.
- FFT, Welch PSD, instantaneous amplitude/phase statistics, higher-order cumulants 같은 feature를 fusion하면 안정성이 좋아질 수 있다.

현재 실험과의 관계:

- 현재는 raw IQ CNN만 사용한다.
- 실험 2에서는 raw IQ CNN을 우선 baseline으로 유지하되, 실패 분석용으로 FFT/Welch/HOC feature branch를 추가 검토한다.

### 1.6 Zhang 2021, Attention + Hybrid Parallel CNN/GRU

핵심:

- CNN은 spatial/local feature, GRU는 temporal feature를 잡는 데 유리하다.
- attention은 feature별 중요도를 재가중하는 데 사용한다.
- AM-Softmax는 class 간 거리를 벌리고 class 내부 거리를 줄이는 목적이다.

현재 실험과의 관계:

- 실험 2의 1차 목표는 데이터셋과 검증 체계를 바로잡는 것이므로 CNN-GRU attention은 2차 모델 후보로 둔다.
- payload/session confounding을 제거한 후에도 CNN이 한계에 걸리면 CNN-GRU/attention을 추가한다.

### 1.7 O'Shea 2017, Semi-Supervised Radio Signal Identification

핵심:

- 실제 radio domain에서는 label이 있는 curated dataset이 부족하다.
- unlabeled capture를 embedding/clustering해서 새 신호를 식별하는 접근이 필요하다.

현재 실험과의 관계:

- 지금은 supervised 3-class AMC 단계다.
- 실험 2 이후에는 background capture와 unknown signal capture를 모아 embedding 시각화/cluster 분석을 추가할 수 있다.

### 1.8 Channel Autoencoder 논문

핵심:

- 통신 자체를 end-to-end autoencoder로 학습할 수 있으며, channel regularizer가 중요하다.
- 128 random bits dataset, impairment layer, attention/regularization 개념이 제시된다.

현재 실험과의 관계:

- 현재 프로젝트의 목표는 learned communication이 아니라 AMC이므로 직접 구현 대상은 아니다.
- 다만 simulation channel augmentation 설계 근거로 사용한다.

---

## 2. 현재 실험 1에 대한 논문 기준 검토

### 2.1 잘 맞는 부분

- I/Q를 2채널로 나누어 CNN에 넣는 구조는 O'Shea 계열 논문과 일치한다.
- raw IQ 기반 feature learning을 사용한다.
- 실제 HackRF TX + RTL-SDR RX를 사용해 OTA capture를 수행했다.
- 수신 IQ에서 window를 추출해 `.npz` sample로 관리한다.
- confusion matrix, precision/recall/F1을 저장한다.

### 2.2 반드시 고쳐야 할 부분

1. **payload-class confounding**
   - 현재 실험 1은 `BASK=A`, `BFSK=F`, `BPSK=P`로 payload가 class마다 고정되어 있다.
   - 모델이 변조 방식이 아니라 payload pattern 또는 반복 frame structure를 함께 학습했을 가능성이 있다.
   - 실험 2에서는 모든 modulation에서 동일 payload pool을 사용해야 한다.

2. **window-level split leakage**
   - 현재 random-window split은 같은 raw capture 안의 window가 train/val/test에 섞인다.
   - 이 경우 모델이 capture session의 gain, offset, noise floor, hardware artifact를 기억할 수 있다.
   - 논문식 OTA 검증으로 보려면 session-level split이 필요하다.

3. **simulation channel이 너무 단순함**
   - 현재 simulation은 AWGN 중심이다.
   - 논문들은 CFO, SRO, phase offset, random gain, multipath/Rayleigh fading, time offset, DC/IQ impairment를 주요 변수로 본다.
   - sim-to-real gap을 줄이려면 channel augmentation을 추가해야 한다.

4. **수신 window의 effective symbol 수가 부족할 수 있음**
   - 현재 real input은 RX 2.4 MS/s에서 1024 sample window다.
   - 시간 길이는 약 `1024 / 2.4e6 = 0.426 ms`다.
   - symbol rate가 5 ksps이면 한 symbol은 0.2 ms이므로 한 window에는 약 2.1 symbol만 들어간다.
   - O'Shea류 window는 짧더라도 modulation feature가 충분히 들어오게 sample/symbol과 window 길이를 맞춘다.
   - 실험 2에서는 channelization/downsampling 또는 larger window 전략을 명시해야 한다.

5. **off-tune/channel selection이 실험 변수로 관리되지 않음**
   - 실험 1에서는 100 kHz baseband tone 주변을 사용했다.
   - O'Shea 2018은 direct-conversion DC impairment를 피하기 위해 약 1 MHz off-tune을 언급한다.
   - 실험 2에서는 channel offset 후보를 sweep하고, 선택 근거를 spectrum 분석으로 남겨야 한다.

---

## 3. 수정된 실험 목표

실험 2의 목표를 다음으로 수정한다.

> 실험 2는 "같은 capture 내부 random-window accuracy"가 아니라, 서로 다른 capture session과 payload/gain/frequency 조건에서도 BASK/BFSK/BPSK 변조 방식이 일반화되어 분류되는지 검증하는 실험이다.

성공 기준:

- session-held-out test accuracy `>= 0.75`
- class별 recall 모두 `>= 0.65`
- BASK/BFSK/BPSK confusion matrix에서 특정 한 class로 collapse하지 않을 것
- payload를 바꿔도 class accuracy가 유지될 것
- SNR/gain/offset 조건별 성능 곡선을 저장할 것

실험 1의 `0.8193` accuracy는 유지하되, 이는 "동일 capture random-window baseline"으로만 해석한다.

---

## 4. 수정된 데이터셋 설계

### 4.1 payload 설계

실험 2부터는 class별 고정 payload를 금지한다.

payload pool:

```text
["A", "F", "P", "0", "1", "7", "K", "R", "S", "Z"]
```

각 modulation은 동일 payload pool에서 payload를 순환 또는 random 선택한다.

예:

```text
session_001/BASK: A, F, P, 0, 1, ...
session_001/BFSK: A, F, P, 0, 1, ...
session_001/BPSK: A, F, P, 0, 1, ...
```

목적:

- label이 payload에 새지 않게 한다.
- 모델이 payload가 아니라 modulation feature를 보도록 만든다.

### 4.2 session split

capture session 단위로 split한다.

권장 최소 구성:

```text
train sessions: 6
val sessions:   2
test sessions:  2
```

각 session은 모든 modulation을 포함한다.

```text
session_001/
  bask_*.bin
  bfsk_*.bin
  bpsk_*.bin
session_002/
  ...
```

split 규칙:

```text
train: session_001 ~ session_006
val:   session_007 ~ session_008
test:  session_009 ~ session_010
```

금지:

- 같은 raw capture에서 뽑은 window를 train/val/test에 동시에 넣지 않는다.

### 4.3 channel offset sweep

실험 2 capture는 baseband carrier offset을 변수로 관리한다.

후보:

```text
100 kHz
250 kHz
500 kHz
```

절차:

1. 각 offset에서 짧은 pilot capture를 수행한다.
2. spectrum peak, DC spike, noise floor를 분석한다.
3. 가장 안정적인 offset을 primary condition으로 선택한다.
4. 나머지 offset은 robustness test로 남긴다.

### 4.4 gain/SNR sweep

HackRF TX VGA gain과 RX gain을 조건으로 기록한다.

기본 후보:

```text
tx_vga_gain: [20, 30, 35]
rx_gain:     [20, 30, 40]
attenuator:  실제 연결 기준 기록
```

각 capture마다 metadata에 다음을 저장한다.

```text
session_id
modulation
payload
center_freq
baseband_offset_hz
tx_sample_rate
rx_sample_rate
symbol_rate
tx_vga_gain
tx_amp_gain
rx_gain
attenuator_db
capture_seconds
tx_seconds
estimated_snr_db
noise_floor
peak_frequency_hz
```

### 4.5 window/sample-rate 설계

현재 방식:

```text
rx_sample_rate = 2.4 MS/s
window_size = 1024
symbol_rate = 5 ksps
window duration = 0.426 ms
symbols/window ≈ 2.1
```

수정 방향:

**우선안 A: channelize + downsample**

1. 수신 IQ를 baseband offset 근처로 frequency shift한다.
2. low-pass filter를 적용한다.
3. `160 kS/s` 또는 `200 kS/s`로 resample한다.
4. `window_size=1024`를 유지한다.

예상:

```text
sample_rate_after_channelize = 160 kS/s
symbol_rate = 5 ksps
samples/symbol = 32
symbols/window = 32
```

장점:

- O'Shea식 1024 input과 호환된다.
- 한 window에 충분한 symbol transition이 들어간다.
- CNN이 단순 carrier fragment가 아니라 modulation behavior를 볼 수 있다.

**대안 B: raw sample rate 유지 + larger window**

```text
rx_sample_rate = 2.4 MS/s
window_size = 16384 or 32768
```

단점:

- 모델 입력이 커지고 학습 비용이 증가한다.
- O'Shea 1024 sample 입력 관례와 멀어진다.

결론:

- 실험 2 기본안은 **channelize + downsample + 1024 window**로 한다.

---

## 5. 수정된 simulation 계획

실험 2 simulation은 AWGN-only에서 벗어나 다음 impairment를 추가한다.

필수 augmentation:

```text
AWGN: SNR -10, -5, 0, 5, 10, 15, 20 dB
random gain scale
random phase offset
carrier frequency offset
sample rate offset 또는 resampling drift
random time offset
DC offset
IQ imbalance
simple multipath/Rayleigh or Rician fading
```

목적:

- O'Shea 2016/2018의 channel realism에 가까워진다.
- real capture에서 발생하는 offset/gain/fading 변화에 덜 취약한 모델을 만든다.

실험 순서:

```text
sim_awgn_only baseline
sim_impairment_augmented
real_session_only
sim_augmented_pretrain + real_finetune
```

비교할 metric:

```text
overall accuracy
macro F1
class recall
condition-wise accuracy by SNR/gain/offset/session
confusion matrix
```

---

## 6. 수정된 모델 계획

### 6.1 Model A: 현재 CNN baseline 유지

현재 모델:

```text
Input [B, 2, 1024]
Conv1d 2→32
Conv1d 32→64
Conv1d 64→128
AdaptiveAvgPool1d
Linear 128→64→3
```

용도:

- 실험 1과 직접 비교하는 baseline.

### 6.2 Model B: O'Shea 2018 VGG-style 1D CNN

논문 참고 구조:

```text
Input 2 × 1024
Conv/Pool 반복
FC/SELU or ReLU
Softmax
```

수정 구현:

- Conv block을 5~6개로 확장
- BatchNorm 유지
- Dropout 또는 AlphaDropout 적용
- Global average pooling 검토

용도:

- raw IQ CNN의 논문식 baseline.

### 6.3 Model C: ResNet1D

O'Shea 2018에서 ResNet 계열이 VGG보다 좋은 성능을 보였다.

구현 조건:

- residual block 3~6개
- channel size 32/64/128
- global average pooling

용도:

- session-held-out에서 CNN baseline이 부족할 때 주력 후보.

### 6.4 Model D: CNN-GRU Attention 후보

Zhang 2021 근거:

- CNN branch: local/spatial feature
- GRU branch: temporal feature
- attention: feature reweighting

도입 조건:

- Model A/B/C에서 session-held-out accuracy가 `0.75` 미만일 때.
- 또는 class별 recall 편차가 큰 경우.

### 6.5 Feature fusion은 2차 분석 도구로 유지

Han 2021 근거로 FFT/Welch/HOC branch를 검토하되, 실험 2의 1차 목표는 데이터셋과 split 개선이다.

도입 순서:

1. FFT/Welch/HOC feature extractor 구현
2. feature-only classifier baseline
3. raw IQ CNN + feature fusion 비교

---

## 7. 수정된 실험 실행 단계

### Phase 0: 실험 1 고정

- 실험 1 브랜치: `exp01-baseline-real-random-window`
- 실험 1 결과는 변경하지 않는다.
- 실험 2는 `experiments/exp02_session_generalization/` 아래에 기록한다.

### Phase 1: capture 전 spectrum survey

목적:

- DC spike, 외부 간섭, 실제 peak 위치 확인.

실행:

```powershell
python -m src.sdr.capture_iq --output data\real\survey\noise_only.bin --seconds 5
python -m src.sdr.analyze_capture --input data\real\survey\noise_only.bin
```

추가:

- TX on pilot capture를 offset별로 수행한다.
- `100k`, `250k`, `500k` offset 중 안정적인 조건을 선택한다.

### Phase 2: session capture 자동화 수정

필요 기능:

- session id 지정
- payload pool 순환
- modulation별 반복 capture
- gain/offset metadata 저장
- noise-only segment 저장

출력 예:

```text
data/real/exp02/raw/session_001/
  metadata.json
  noise_only.bin
  BASK_payload_A_offset250k_gain30.bin
  BFSK_payload_A_offset250k_gain30.bin
  BPSK_payload_A_offset250k_gain30.bin
  ...
```

### Phase 3: channelize + downsample import

필요 기능:

- raw IQ load
- frequency shift by `baseband_offset_hz`
- low-pass FIR
- resample to `160 kS/s` 또는 `200 kS/s`
- normalize
- 1024 sample window extraction
- session-level split

출력:

```text
data/real/exp02/processed/train/
data/real/exp02/processed/val/
data/real/exp02/processed/test/
manifest_exp02.json
```

### Phase 4: baseline 학습

순서:

```text
1. current CNN on exp02
2. VGG-style CNN on exp02
3. ResNet1D on exp02
4. sim_augmented_pretrain + real_finetune
```

### Phase 5: 논문식 평가

필수 결과:

```text
overall accuracy
macro precision/recall/F1
confusion matrix
accuracy by session
accuracy by payload
accuracy by tx_gain/rx_gain
accuracy by baseband_offset_hz
accuracy by estimated_snr_db bin
```

저장 위치:

```text
experiments/exp02_session_generalization/results/
  metrics.json
  confusion_matrix.png
  accuracy_by_session.csv
  accuracy_by_payload.csv
  accuracy_by_gain.csv
  accuracy_by_offset.csv
  accuracy_by_snr.csv
```

---

## 8. 현재 계획에서 즉시 수정할 사항

우선순위 P0:

- payload 고정 제거
- session-level split 적용
- metadata schema 확장
- channelize/downsample import 추가
- 실험 2 폴더를 기본 output-root로 사용

우선순위 P1:

- simulation impairment augmentation 추가
- VGG-style CNN, ResNet1D 추가
- condition-wise evaluation 추가

우선순위 P2:

- CNN-GRU attention
- FFT/Welch/HOC feature fusion
- semi-supervised embedding/clustering

---

## 9. 결론

현재 실험 1은 "실제 SDR capture를 사용해 CNN 기반 AMC가 가능하다"는 좋은 출발점이다. 그러나 논문 기준으로는 아직 일반화 실험이 아니다. 특히 payload-class confounding과 같은-session random-window split이 가장 큰 약점이다.

따라서 실험 2는 모델을 복잡하게 만드는 것보다 먼저 데이터셋과 평가 프로토콜을 바로잡아야 한다. 논문 기반 수정 계획의 핵심은 다음 세 문장으로 요약된다.

1. 모든 modulation에서 같은 payload pool을 사용한다.
2. train/val/test는 반드시 capture session 단위로 분리한다.
3. real capture는 channelize/downsample하여 한 window에 충분한 symbol transition이 들어가게 한다.

이 세 가지가 반영된 뒤에 VGG/ResNet/Attention/Feature Fusion을 비교해야 논문식으로 의미 있는 실험 결과가 된다.
