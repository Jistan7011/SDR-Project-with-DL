# Experiment 1 Plan

## 목적

HackRF One으로 BASK/BFSK/BPSK 신호를 송신하고 RTL-SDR Blog V4로 수신한 real IQ capture를 이용해 1D CNN 변조 분류 baseline을 확보한다.

## 실험 정의

```text
name: exp1
status: completed baseline
TX: HackRF One
RX: RTL-SDR Blog V4
center_freq: 433 MHz
tx_sample_rate: 8 MS/s
rx_sample_rate: 2.4 MS/s
symbol_rate: 5 ksps
tx_vga_gain: 30
rf_path: OTA antenna
tx_rx_distance: about 10 cm
antenna_layout: side_by_side
window_size: 1024
input_shape: [batch, 2, 1024]
split_policy: same-capture random window split
```

실험 1은 동축 케이블/감쇠기 직결 실험이 아니라, HackRF One과 RTL-SDR V4에 각각
안테나를 연결한 뒤 약 `10 cm` 거리에서 나란히 세워 수행한 근거리 OTA 실험이다.

## 데이터 설계

실험 1에서는 class별 payload를 고정했다.

```text
BASK -> payload "A"
BFSK -> payload "F"
BPSK -> payload "P"
```

각 payload는 1 byte 문자이며 frame은 총 40 bits다.

```text
preamble 16 bits
sync word 8 bits
payload 8 bits
CRC8 8 bits
```

## 학습 설계

수신 IQ는 complex64 stream이다. 각 window를 평균 제거/표준편차 정규화한 뒤 I/Q를 2채널로 나눠 CNN에 넣는다.

```text
single sample: [2, 1024]
batch: [B, 2, 1024]
label: BASK/BFSK/BPSK
```

## 성공 기준

실험 1은 `10 cm 근거리 OTA real capture`에서 end-to-end pipeline이 동작하는지 보는 baseline이다. 기준 결과는 real random-window dataset에서 accuracy 약 0.8193이다.

## 한계

같은 raw capture 안의 windows가 train/val/test에 섞인다. 또한 TX/RX 안테나 거리가 약 `10 cm`인 근거리 OTA 조건만 사용했다. 따라서 session 일반화 또는 거리 일반화 성능으로 해석하면 안 된다. 이 한계를 해결하기 위해 실험 2를 별도 폴더에서 진행한다.
