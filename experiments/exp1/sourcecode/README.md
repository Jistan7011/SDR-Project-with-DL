# SDR Modulation AI

PyTorch 1D CNN 기반 BASK/BFSK/BPSK 변조 분류와 DSP 복조 실험 프로젝트입니다.

## Environment

Use Python 3.11 and a project-local virtual environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Start

```powershell
.\.venv\Scripts\python.exe -m src.dataset.generate_sim_dataset --preset smoke
.\.venv\Scripts\python.exe -m src.train.train_cnn1d --config config.yaml --preset smoke
.\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint results/checkpoints/best.pt
.\.venv\Scripts\python.exe -m src.app.decode_iq --input data/sim/test/sample_000000.npz --checkpoint results/checkpoints/best.pt
```

## SDR Commands

Use Radioconda for SDR commands because SoapySDR and the hardware modules live there.
Open a Radioconda Prompt, or activate it from `cmd`:

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.capture_iq --output data\real\raw_iq\capture.bin --seconds 3"
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.hackrf_tx --modulation BPSK --payload P --seconds 3"
```

Directly calling `C:\Users\qus70\radioconda\python.exe` is not enough on Windows; the Radioconda activation step sets the DLL search path needed by SoapySDR modules.

## Import Real Captures

After capturing raw `complex64` files, convert them into labeled `.npz` windows:

```powershell
.\.venv\Scripts\python.exe -m src.dataset.import_real_iq --input data\real\raw_iq\bask_a.bin --modulation BASK --payload A --skip-seconds 0.5 --duration-seconds 3 --max-windows 300
.\.venv\Scripts\python.exe -m src.dataset.import_real_iq --input data\real\raw_iq\bfsk_f.bin --modulation BFSK --payload F --skip-seconds 0.5 --duration-seconds 3 --max-windows 300
.\.venv\Scripts\python.exe -m src.dataset.import_real_iq --input data\real\raw_iq\bpsk_p.bin --modulation BPSK --payload P --skip-seconds 0.5 --duration-seconds 3 --max-windows 300
.\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint results/checkpoints/best.pt --data-root data/real/processed
```

For real fine-tuning, prefer randomized windows across the full active TX interval:

```powershell
Remove-Item -Recurse -Force data\real\processed
.\.venv\Scripts\python.exe -m src.dataset.import_real_sequence --windows-per-class 900
.\.venv\Scripts\python.exe -m src.train.train_cnn1d --config config.yaml --preset train
.\.venv\Scripts\python.exe -m src.train.evaluate --checkpoint results/checkpoints/best.pt --data-root data/real/processed
```

Analyze raw captures before importing when real accuracy is poor:

```powershell
.\.venv\Scripts\python.exe -m src.sdr.analyze_capture --input data\real\raw_iq\bask_a.bin
.\.venv\Scripts\python.exe -m src.sdr.analyze_capture --input data\real\raw_iq\bfsk_f.bin
.\.venv\Scripts\python.exe -m src.sdr.analyze_capture --input data\real\raw_iq\bpsk_p.bin
```

## Automated SDR Capture Sequence

Run RTL-SDR capture and HackRF TX in a timed sequence for BASK/A, BFSK/F, and BPSK/P:

```powershell
.\.venv\Scripts\python.exe -m src.experiment.run_sdr_capture_sequence --import-after
```

This starts RX first, waits 1 second, starts TX for 3 seconds, waits for the 8-second capture to finish, then imports the result into `data/real/processed/test`.

Preview commands without touching hardware:

```powershell
.\.venv\Scripts\python.exe -m src.experiment.run_sdr_capture_sequence --dry-run
```
