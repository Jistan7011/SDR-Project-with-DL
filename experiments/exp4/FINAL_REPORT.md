# 실험 4 최종 보고서

## 요약

실험 4의 목적은 기존 실험 1~3의 `IQ window -> 딥러닝 변조 분류`를 확장해, 실제 수신 IQ에서 payload 복원까지 이어지는 end-to-end 파이프라인을 검증하는 것이다.

검증한 전체 흐름은 다음과 같다.

```text
RTL-SDR 수신 IQ
-> 딥러닝 변조 분류기
-> 예측된 변조 방식에 맞는 DSP 복조기 선택
-> frame sync
-> payload 복원
-> CRC / BER / CER / Packet Success Rate 평가
```

실험 4는 허가된 자체 송신 신호만 사용했다. AWGN을 의도적으로 섞거나 다중 신호를 동시에 전송하는 실험은 포함하지 않았다.

## RF 실험 환경

```text
TX: HackRF One
RX: RTL-SDR Blog V4
RF path: OTA antenna
SDR 간 RF 케이블 연결: 없음
TX/RX 거리: 약 1 m
안테나 배치: face_to_face
center frequency: 433 MHz
RX sample rate: 2.4 MS/s
TX sample rate: 8 MS/s
symbol rate: 5 ksps
payload pool: ["A", "F", "P", "0", "1", "7", "K", "R", "S", "Z"]
modulation classes: ["BASK", "BFSK", "BPSK"]
```

## 데이터셋 구성

공식 실험 4는 15개 session으로 구성했다.

```text
sessions: 15
captures per session: 3 modulations x 10 payloads = 30
total OTA captures: 450
frame windows per capture: 20
processed samples: 9000
```

split은 session-held-out 방식으로 구성했다.

```text
train: session_001 ~ session_009 = 5400 samples
val:   session_010 ~ session_012 = 1800 samples
test:  session_013 ~ session_015 = 1800 samples
```

같은 session에서 나온 sample이 train/val/test에 동시에 들어가지 않도록 확인했다. session leakage는 발견되지 않았다.

## Frame 구조

각 payload는 1 byte 문자이며, frame은 다음 구조를 사용한다.

```text
preamble 16 bits: 1010101010101010
sync word 8 bits: 11001100
payload 8 bits
CRC8 8 bits
----------------
total 40 bits
```

CRC8은 polynomial `0x07`, init `0x00`을 사용한다.

## 모델

분류기는 실험 3의 best ResNet1D checkpoint에서 초기화한 뒤, 실험 4 train session으로 fine-tuning했다.

```text
model: ResNet1D
input feature: [I, Q, instantaneous_frequency]
input shape: [batch, 3, 1024]
output classes: ["BASK", "BFSK", "BPSK"]
```

사용한 checkpoint:

```text
results/exp04_resnet_finetune/checkpoints/best.pt
```

best validation accuracy는 약 `0.7306`이었다.

## 변조 분류 결과

held-out test session 기준 분류 성능은 다음과 같다.

```text
classification accuracy: 0.7178
macro F1: 0.7189
```

class별 recall:

```text
BASK recall: 0.8483
BFSK recall: 0.6683
BPSK recall: 0.6367
```

confusion matrix:

```text
          predicted
true      BASK  BFSK  BPSK
BASK       509    62    29
BFSK       176   401    23
BPSK       145    73   382
```

해석하면 BASK는 비교적 잘 분류되지만, BFSK와 BPSK 일부가 BASK로 잘못 분류되는 경향이 있다.

## Payload 복원 결과

held-out test session 기준 end-to-end recovery 결과는 다음과 같다.

```text
samples: 1800
classification_accuracy: 0.7178
crc_pass_rate: 0.7028
packet_success_rate: 0.7022
payload_recovery_accuracy: 0.7022
oracle_crc_pass_rate: 0.9778
mean_ber: 0.2928
mean_cer: 0.2978
```

여기서 `oracle_crc_pass_rate`는 정답 modulation을 알고 있다고 가정하고 DSP 복조기를 선택했을 때의 CRC 통과율이다. 따라서 이 값은 DSP 복조기와 frame recovery 자체의 성능을 classifier 오류와 분리해서 보여준다.

```text
AI 예측 modulation 기준 packet success: 0.7022
정답 modulation 기준 oracle CRC pass: 0.9778
```

즉, 변조 방식만 맞게 선택되면 payload 복원은 대부분 성공한다. 현재 전체 성능을 제한하는 가장 큰 요인은 DSP 복조기 자체보다 변조 분류 오류다.

## 변조별 복원 결과

```text
BASK packet success: 0.8217
BFSK packet success: 0.6683
BPSK packet success: 0.6167
```

변조별 oracle CRC pass rate:

```text
BASK oracle CRC pass: 0.9667
BFSK oracle CRC pass: 1.0000
BPSK oracle CRC pass: 0.9667
```

BPSK는 초기에는 복원이 거의 되지 않았지만, DSP 복조 보강 후 oracle 기준으로는 BASK/BFSK와 비슷한 수준까지 회복됐다.

## Session별 복원 결과

```text
session_013 packet success: 0.7133
session_014 packet success: 0.7033
session_015 packet success: 0.6900
```

test session 3개 사이의 성능 차이는 크지 않았다. session_014는 baseband offset `250 kHz` 조건을 포함했지만, 전체 packet success가 다른 test session과 크게 다르지 않았다.

## Failure Stage 분석

실패 단계는 다음과 같이 분리했다.

```text
success: 1264
classification failure: 508
frame_sync failure: 28
```

전체 1800개 test sample 중 1264개는 payload와 CRC를 모두 성공적으로 복원했다. 실패의 대부분은 modulation classification 오류였다.

## 시행착오 및 수정

초기 recovery 평가에서는 BPSK의 oracle CRC pass rate가 `0.0`이었다. 이는 단순히 classifier가 틀린 문제가 아니라, BPSK DSP 복조기 자체가 OTA phase drift에 취약하다는 뜻이었다.

처음 BPSK 복조기는 고정 phase 기준 coherent demodulation에 가까운 방식이었다. 하지만 실제 OTA 환경에서는 수신 시작 시점, LO phase, phase drift 때문에 고정 phase 기준이 쉽게 무너진다.

이를 해결하기 위해 BPSK recovery 경로에 differential BPSK 후보를 추가했다.

```text
symbol mean extraction
-> adjacent-symbol phase transition detection
-> transition sequence로 candidate bits 재구성
-> preamble/sync/CRC 기준으로 best candidate 선택
```

수정 전후 결과:

```text
payload recovery accuracy: 0.4967 -> 0.7022
oracle CRC pass rate: 0.6556 -> 0.9778
```

이 결과는 실험 4의 중요한 발견이다. end-to-end 시스템의 병목은 두 가지였다.

```text
1. classifier가 BFSK/BPSK를 BASK로 잘못 분류하는 문제
2. BPSK 복조기의 phase drift 취약성
```

2번은 differential recovery 보강으로 대부분 해결됐다. 남은 주요 병목은 1번, 즉 classifier bias다.

## 성공 기준 평가

실험 4 계획의 최소 성공 기준:

```text
classification accuracy >= 0.70
결과: 0.7178, 통과

oracle-demod CRC pass rate >= 0.60
결과: 0.9778, 통과

predicted-modulation payload recovery accuracy >= 0.40
결과: 0.7022, 통과
```

목표 성공 기준:

```text
classification accuracy >= 0.75
결과: 0.7178, 미달

oracle-demod CRC pass rate >= 0.75
결과: 0.9778, 통과

predicted-modulation payload recovery accuracy >= 0.60
결과: 0.7022, 통과
```

따라서 실험 4는 payload recovery 관점에서는 성공으로 판단한다. 다만 classification accuracy 목표 `0.75`에는 도달하지 못했으므로, 다음 실험에서는 classifier 개선이 필요하다.

## 결론

실험 4의 결론은 다음과 같다.

```text
1 m OTA 환경에서 딥러닝 변조 분류 + DSP 복조 기반 payload 복원 파이프라인은 동작한다.

정답 modulation을 알고 있으면 DSP 복조와 frame recovery는 거의 성공한다.

AI 예측 modulation을 사용할 때도 payload recovery accuracy 0.7022를 달성했다.

남은 핵심 병목은 DSP 복조가 아니라 BFSK/BPSK를 BASK로 잘못 분류하는 classifier bias다.
```

즉, 실험 4는 “분류 결과가 실제 데이터 복원으로 이어질 수 있는가?”라는 질문에 대해 **가능하다**는 답을 준다. 동시에 다음 단계에서는 modulation classifier의 robustness와 calibration을 개선해야 함을 보여준다.

## 산출물

주요 산출물은 다음 위치에 저장되어 있다.

```text
results/exp04_resnet_finetune/checkpoints/best.pt
results/exp04_resnet_finetune/logs/eval_metrics.json
results/reports/recovery_eval_finetuned_test/recovery_eval.json
results/reports/recovery_eval_finetuned_test/recovery_eval.csv
results/reports/recovery_summary.md
results/reports/recovery_metrics.json
results/reports/recovery_by_modulation.csv
results/reports/recovery_by_payload.csv
results/reports/recovery_by_session.csv
results/reports/confusion_matrix.png
results/reports/crc_packet_success.png
```

## 다음 실험 방향

다음 실험은 payload DSP recovery보다 classifier robustness 개선에 집중하는 것이 타당하다.

권장 방향:

```text
Experiment 5:
  SNR / gain / offset stress test
  confidence thresholding
  reject / unknown option
  BASK over-prediction 감소를 위한 class calibration
  BFSK/BPSK recall 개선
```

실험 5에서는 단순 accuracy뿐 아니라 confidence, false alarm, unknown/reject 처리를 함께 봐야 한다.
