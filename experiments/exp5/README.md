# Experiment 5: BFSK/BPSK 분류 성능 개선

실험 5는 exp4에서 확인된 병목인 `BFSK/BPSK -> BASK` 오분류를 줄이는 실험이다. 복원 파이프라인보다 분류기 성능을 우선하며, 모델 선택 기준은 overall accuracy가 아니라 `BFSK recall`, `BPSK recall`, `worst-class recall`이다.

## 핵심 조건

- RF path: OTA antenna
- RF cable between SDR: false
- TX: HackRF One
- RX: RTL-SDR Blog V4
- 거리: 약 1 m
- center frequency: 433 MHz
- TX sample rate: 8 MS/s
- RX sample rate: 2.4 MS/s
- symbol rate: 5 ksps
- payload pool: `A,F,P,0,1,7,K,R,S,Z`
- 기본 window: `2048`
- 기본 feature: `[I,Q,magnitude,instantaneous_frequency,differential_phase]`

## 1. 환경 확인

```powershell
cd D:\ai_projects\SDR\experiments\exp5\sourcecode

..\..\..\.venv\Scripts\python.exe -m pytest -q ..\tests

cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

## 2. 신규 session dry-run

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_016 --tx-vga-gain 25 --rx-gain 30 --baseband-offset-hz 500000 --dry-run
```

## 3. 신규 15 session 수집

아래 명령은 session별로 BASK/BFSK/BPSK x payload 10개를 순차 송수신한다. 각 session은 약 30개 capture와 noise-only capture 1개를 만든다.

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_016 --tx-vga-gain 25 --rx-gain 30 --baseband-offset-hz 500000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_017 --tx-vga-gain 30 --rx-gain 30 --baseband-offset-hz 500000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_018 --tx-vga-gain 35 --rx-gain 30 --baseband-offset-hz 500000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_019 --tx-vga-gain 30 --rx-gain 25 --baseband-offset-hz 500000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_020 --tx-vga-gain 30 --rx-gain 35 --baseband-offset-hz 500000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_021 --tx-vga-gain 25 --rx-gain 30 --baseband-offset-hz 250000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_022 --tx-vga-gain 30 --rx-gain 30 --baseband-offset-hz 250000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_023 --tx-vga-gain 35 --rx-gain 30 --baseband-offset-hz 250000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_024 --tx-vga-gain 30 --rx-gain 25 --baseband-offset-hz 250000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_025 --tx-vga-gain 30 --rx-gain 35 --baseband-offset-hz 250000 --antenna-note face_to_face_center
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_026 --tx-vga-gain 32 --rx-gain 28 --baseband-offset-hz 500000 --antenna-note slight_angle_15deg
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_027 --tx-vga-gain 28 --rx-gain 32 --baseband-offset-hz 500000 --antenna-note slight_angle_15deg
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_028 --tx-vga-gain 32 --rx-gain 28 --baseband-offset-hz 250000 --antenna-note slight_angle_15deg
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_029 --tx-vga-gain 28 --rx-gain 32 --baseband-offset-hz 250000 --antenna-note slight_angle_15deg
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_exp05_capture_session --session-id session_030 --tx-vga-gain 30 --rx-gain 30 --baseband-offset-hz 500000 --antenna-note height_offset_small
```

## 4. Classification dataset import

exp4 raw IQ는 복사하지 않고 read-only로 참조한다.

```powershell
Remove-Item -Recurse -Force ..\data\processed -ErrorAction SilentlyContinue

..\..\..\.venv\Scripts\python.exe -m src.dataset.import_exp05_classification_windows --config ..\config\config.exp05.yaml --raw-roots ..\..\exp4\data\raw_iq ..\data\raw_iq --output-root ..\data\processed
```

## 5. 학습

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp05.yaml --data-root ..\data\processed --output-root ..\results\exp05_resnet_2048 --model-type resnet1d --eval-split test --seeds 42 43 44

..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp05.yaml --data-root ..\data\processed --output-root ..\results\exp05_fusion_resnet_2048 --model-type fusion_resnet1d --eval-split test --seeds 42 43 44
```

## 6. Class bias calibration

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.calibrate_exp05_class_bias --checkpoint ..\results\exp05_resnet_2048\resnet1d_seed42\checkpoints\best.pt --config ..\config\config.exp05.yaml --data-root ..\data\processed --output-dir ..\results\exp05_resnet_2048\resnet1d_seed42\calibration --objective worst_recall --eval-splits val test
```

## 7. 오류 분석

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_exp05_bfsk_bpsk_errors --metrics ..\results\exp05_resnet_2048\resnet1d_seed42\logs\eval_metrics.json --output-dir ..\results\analysis\resnet_seed42
```

## 성공 기준

- overall accuracy >= 0.78
- BFSK recall >= 0.75
- BPSK recall >= 0.75
- worst-class recall >= 0.74
- BASK recall >= 0.75
- BFSK/BPSK -> BASK 오분류율이 exp4 대비 35% 이상 감소
