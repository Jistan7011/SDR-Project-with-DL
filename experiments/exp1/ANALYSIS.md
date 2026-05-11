# Experiment 1 Analysis

## 결과 요약

실험 1은 HackRF One TX와 RTL-SDR V4 RX를 사용해 실제 SDR capture 기반 BASK/BFSK/BPSK 3-class 분류 baseline을 만들었다.

기준 결과:

```text
accuracy: 약 0.8193
macro F1: 약 0.8189
```

## 시행착오

시뮬레이션 데이터만으로 학습한 모델은 실제 SDR capture에서 accuracy가 약 0.3333 수준이었다. 이는 3-class random guess와 거의 같아서 sim-to-real gap이 컸다고 해석했다.

수동 송수신은 RX/TX 시작 타이밍 편차가 커서 자동화 스크립트로 바꿨다.

TX gain 0에서는 신호가 약했고, `--tx-vga-gain 30`에서 성능이 개선되었다. `--tx-vga-gain 40`은 더 좋은 결과로 이어지지 않았다.

시간 구간을 train/val/test로 나누는 방식은 train loss가 낮아도 val accuracy가 0.3333 근처에 머무는 문제가 있었다.

최종적으로 송신 활성 구간에서 random window를 뽑아 train/val/test로 나누는 방식에서 accuracy 약 0.8193을 얻었다.

## 한계

실험 1은 같은 capture 안에서 random window를 나눈다. 따라서 모델이 modulation feature뿐 아니라 같은 session의 gain, offset, noise floor, hardware artifact를 함께 기억했을 수 있다.

또한 `BASK=A`, `BFSK=F`, `BPSK=P`로 payload가 class마다 고정되어 payload-class confounding 가능성이 있다.

## 다음 실험으로 넘길 내용

실험 2에서는 다음을 반드시 분리한다.

- 모든 modulation에서 동일 payload pool 사용
- train/val/test를 session 단위로 분리
- noise-only capture와 estimated SNR metadata 저장
- channelize/downsample 후 `[2, 1024]` window 생성

## 구조 변경 후 검증

프로젝트를 `experiments/exp1`, `experiments/exp2` 중심 구조로 바꾼 뒤 실험 1을 다시 점검했다. 자세한 명령과 결과는 `TEST_REPORT.md`에 기록했다.

검증 결과:

```text
unit tests: 4 passed
processed dataset: train 3150 / val 675 / test 675
checkpoint evaluation: accuracy=0.8193
actual SDR capture: PASS
actual capture import: train 630 / val 135 / test 135
existing checkpoint on new capture: accuracy=0.3704
```

해석:

구조 변경 후 실험 1의 실제 송수신 파이프라인은 동작한다. 하지만 새로 캡처한 session에 기존 checkpoint를 적용하면 accuracy가 낮다. 이는 실험 1의 random-window baseline이 새 session 일반화를 보장하지 않는다는 뜻이며, 실험 2의 session-held-out 설계가 필요하다는 근거다.
