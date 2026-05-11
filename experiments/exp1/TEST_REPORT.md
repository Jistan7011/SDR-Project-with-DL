# Experiment 1 Structure Test Report

작성일: 2026-05-04

## 목적

프로젝트 구조를 `experiments/exp1`, `experiments/exp2` 중심으로 바꾼 뒤, 실험 1이 독립 폴더 안에서 계속 실행 가능한지 확인했다.

처음에는 RF 송신 없이 구조 검증을 수행했고, 이후 실제 HackRF TX + RTL-SDR RX 송수신까지 수행했다.

- `experiments/exp1/sourcecode` 기준 Python import 정상 여부
- 실험 1 단위 테스트 정상 여부
- 실험 1 processed dataset 접근 여부
- 보존 checkpoint 평가 결과가 기존 기준값과 일치하는지
- decode CLI가 새 경로 구조에서 실행되는지
- SDR capture 명령이 exp1 내부 경로에서 실제 송수신 capture를 수행하는지

## 현재 실험 1 기준 구조

```text
experiments/exp1/
  sourcecode/
    src/
  tests/
    root_tests_snapshot/
  config/
    config.exp01.yaml
  data/
    raw_iq/
    processed/
  results/
    checkpoints/
    logs/
    confusion_matrices/
```

실행 위치는 항상 다음으로 둔다.

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode
```

가상환경 Python은 루트의 `.venv`를 상대 경로로 호출한다.

```powershell
..\..\..\.venv\Scripts\python.exe
```

## 1. 단위 테스트

명령:

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

..\..\..\.venv\Scripts\python.exe -m pytest ..\tests\root_tests_snapshot -q
```

결과:

```text
4 passed in 0.07s
```

판정:

```text
PASS
```

## 2. Processed dataset 확인

명령:

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

Get-ChildItem ..\data\processed -Directory | ForEach-Object {
  [pscustomobject]@{
    Split=$_.Name
    Count=(Get-ChildItem $_.FullName -Filter *.npz -File).Count
  }
} | Format-Table -AutoSize
```

결과:

```text
Split Count
----- -----
test    675
train  3150
val     675
```

판정:

```text
PASS
```

## 3. Checkpoint 평가

명령:

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

..\..\..\.venv\Scripts\python.exe -m src.train.evaluate `
  --checkpoint ..\results\checkpoints\best.pt `
  --config ..\config\config.exp01.yaml `
  --data-root ..\data\processed
```

결과:

```text
accuracy=0.8193
```

판정:

```text
PASS
```

해석:

구조 변경 후에도 실험 1 baseline checkpoint와 processed dataset이 정상 연결된다. 기존 기준 결과인 accuracy 약 `0.8193`과 일치한다.

## 4. Decode CLI 확인

명령:

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

..\..\..\.venv\Scripts\python.exe -m src.app.decode_iq `
  --input ..\data\processed\test\bask_a_001275.npz `
  --checkpoint ..\results\checkpoints\best.pt
```

결과:

```text
{
  'predicted_modulation': 'BFSK',
  'confidence': 0.4431358277797699,
  'recovered': {'payload': '', 'crc_ok': False, 'start': -1, 'bit_count': 2},
  'metadata_assisted_recovery': {'payload': '', 'crc_ok': False, 'start': -1, 'bit_count': 2, 'modulation': 'BASK'}
}
```

판정:

```text
CLI execution PASS
single-sample prediction FAIL
```

해석:

decode CLI는 새 디렉터리 구조에서 정상 실행된다. 다만 선택한 단일 sample은 low confidence로 오분류되었다. 전체 test set 평가 accuracy가 `0.8193`이므로 구조 문제는 아니며, 단일 window 단위에서는 오분류가 발생할 수 있다.

## 5. SDR 장비 인식

명령:

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

결과 요약:

```text
SoapySDR devices:
  HackRF One
  RTL-SDR Blog V4
open driver=rtlsdr: ok
open driver=hackrf: ok
```

판정:

```text
PASS
```

## 6. 실제 SDR 송수신 capture

처음 실행 시 `run_sdr_capture_sequence` 내부에서 `capture_iq`와 `hackrf_tx`가 기본 `config.yaml`을 찾는 문제가 발생했다.

오류:

```text
FileNotFoundError: No such file or directory: 'config.yaml'
```

원인:

프로젝트 구조를 바꾼 뒤 실험 1 config 위치가 `..\config\config.exp01.yaml`로 바뀌었지만, 자동 송수신 스크립트가 하위 모듈 호출 시 config 경로를 전달하지 않았다.

수정:

`experiments/exp1/sourcecode/src/experiment/run_sdr_capture_sequence.py`에서 `capture_iq`, `hackrf_tx`, optional import 호출에 `--config ..\config\config.exp01.yaml`이 전달되도록 수정했다.

명령:

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

..\..\..\.venv\Scripts\python.exe -m src.experiment.run_sdr_capture_sequence `
  --output-dir ..\data\raw_iq_actual_test `
  --config ..\config\config.exp01.yaml `
  --tx-vga-gain 30 `
  --tx-seconds 5 `
  --capture-seconds 10
```

결과:

```text
Captured 24000000 complex64 samples to ..\data\raw_iq_actual_test\bask_a.bin
Captured 24000000 complex64 samples to ..\data\raw_iq_actual_test\bfsk_f.bin
Captured 24000000 complex64 samples to ..\data\raw_iq_actual_test\bpsk_p.bin
```

파일 크기:

```text
bask_a.bin  192000000 bytes
bfsk_f.bin  192000000 bytes
bpsk_p.bin  192000000 bytes
```

판정:

```text
PASS
```

## 7. 실제 송수신 capture import

명령:

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

Remove-Item -Recurse -Force ..\data\processed_actual_test

..\..\..\.venv\Scripts\python.exe -m src.dataset.import_real_sequence `
  --raw-dir ..\data\raw_iq_actual_test `
  --output-root ..\data\processed_actual_test `
  --config ..\config\config.exp01.yaml `
  --windows-per-class 300 `
  --active-start-seconds 1.1 `
  --active-duration-seconds 4.5
```

결과:

```text
Imported real sequence to ..\data\processed_actual_test:
train 630
val   135
test  135
```

판정:

```text
PASS
```

## 8. 새 실제 capture를 기존 checkpoint로 평가

명령:

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

..\..\..\.venv\Scripts\python.exe -m src.train.evaluate `
  --checkpoint ..\results\checkpoints\best.pt `
  --config ..\config\config.exp01.yaml `
  --data-root ..\data\processed_actual_test
```

결과:

```text
accuracy=0.3704
```

판정:

```text
pipeline execution PASS
new-session classifier performance LOW
```

해석:

실제 송수신, raw IQ 저장, processed dataset import, checkpoint 평가 실행은 모두 정상 동작했다. 다만 기존 실험 1 checkpoint를 새로 캡처한 세션에 바로 적용하면 accuracy가 `0.3704`로 낮다.

이는 실험 1이 같은 capture 내부 random-window baseline이라는 한계를 다시 보여준다. 실험 1 baseline `0.8193`은 보존된 같은 조건 dataset에서는 재현되지만, 새 RF capture session 일반화 성능으로 해석하면 안 된다. 이 문제를 해결하기 위해 실험 2에서 session-held-out protocol을 진행한다.

## 재현 명령 전체 순서

아래 명령은 실험 1을 새 구조에서 다시 수행할 때의 표준 순서다. 기존 baseline을 보존하려면 실제 재캡처 산출물은 `raw_iq_actual_test`, `processed_actual_test`처럼 별도 폴더에 둔다.

```powershell
cd D:\ai_projects\SDR\experiments\exp1\sourcecode

..\..\..\.venv\Scripts\python.exe -m pytest ..\tests\root_tests_snapshot -q
```

장비 확인:

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

캡처:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.experiment.run_sdr_capture_sequence `
  --output-dir ..\data\raw_iq_actual_test `
  --config ..\config\config.exp01.yaml `
  --tx-vga-gain 30 `
  --tx-seconds 5 `
  --capture-seconds 10
```

dataset import:

```powershell
Remove-Item -Recurse -Force ..\data\processed_actual_test

..\..\..\.venv\Scripts\python.exe -m src.dataset.import_real_sequence `
  --raw-dir ..\data\raw_iq_actual_test `
  --output-root ..\data\processed_actual_test `
  --config ..\config\config.exp01.yaml `
  --windows-per-class 300 `
  --active-start-seconds 1.1 `
  --active-duration-seconds 4.5
```

학습:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.train_cnn1d `
  --config ..\config\config.exp01.yaml `
  --preset train
```

평가:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.evaluate `
  --checkpoint ..\results\checkpoints\best.pt `
  --config ..\config\config.exp01.yaml `
  --data-root ..\data\processed_actual_test
```

## 최종 판정

```text
Experiment 1 after directory restructure: PASS
Actual HackRF TX + RTL-SDR RX capture: PASS
Existing Exp1 checkpoint on new capture session: LOW accuracy, 0.3704
```

실험 1은 `experiments/exp1` 내부 구조에서 실제 송수신까지 정상 동작한다. 다만 새 capture session에 대한 일반화 성능은 낮으므로, 실험 1은 baseline 보존용으로 유지하고 실험 2에서 session-held-out 검증을 진행해야 한다.
