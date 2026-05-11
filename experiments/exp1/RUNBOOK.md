# Experiment 1 Runbook

실험 1은 `experiments/exp1/` 내부에서 독립적으로 실행한다. 명령은 `sourcecode/`를 현재 위치로 두고 실행한다.

## 1. 환경 확인

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

..\..\..\.venv\Scripts\python.exe --version

..\..\..\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

## 2. SDR capture

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_sdr_capture_sequence --output-dir ..\data\raw_iq --tx-vga-gain 30 --tx-seconds 5 --capture-seconds 10
```

baseline raw IQ를 덮어쓰지 않는 실제 테스트는 다음처럼 별도 폴더에 저장한다.

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_sdr_capture_sequence --output-dir ..\data\raw_iq_actual_test --config ..\config\config.exp01.yaml --tx-vga-gain 30 --tx-seconds 5 --capture-seconds 10
```

생성 파일:

```text
experiments/exp1/data/raw_iq/bask_a.bin
experiments/exp1/data/raw_iq/bfsk_f.bin
experiments/exp1/data/raw_iq/bpsk_p.bin
```

## 3. Dataset import

```powershell
Remove-Item -Recurse -Force ..\data\processed

..\..\..\.venv\Scripts\python.exe -m src.dataset.import_real_sequence --raw-dir ..\data\raw_iq --output-root ..\data\processed --config ..\config\config.exp01.yaml --windows-per-class 1500 --active-start-seconds 1.1 --active-duration-seconds 4.5
```

## 4. 학습

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.train_cnn1d --config ..\config\config.exp01.yaml --preset train
```

## 5. 평가

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint ..\results\checkpoints\best.pt --config ..\config\config.exp01.yaml --data-root ..\data\processed
```

기준 결과:

```text
accuracy ~= 0.8193
```
