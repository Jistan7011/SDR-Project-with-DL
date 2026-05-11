# 실험 2: Session-Held-Out SDR 변조 분류

이 폴더는 실험 2를 독립적으로 재현하기 위한 실험 패키지입니다.

먼저 읽을 문서:

```text
D:\ai_projects\SDR\experiments\exp2\FINAL_REPORT_EXP02.md
```

문서 역할:

```text
README.md             빠른 시작 문서
PLAN.md               실험 2에서 실제로 따른 단일 실험 계획
FINAL_REPORT_EXP02.md 최종 보고서, 분석, 그림, 재현 명령어
```

별도 `RUNBOOK.md`는 두지 않습니다. 실행 명령어는 이 README와 `FINAL_REPORT_EXP02.md`에 정리합니다.

## 1분 요약

실험 2는 실제 SDR IQ 데이터에서 BASK, BFSK, BPSK 변조 방식을 분류하는 모델이 `session-held-out` 조건에서도 일반화되는지 확인한 실험입니다.

중요 RF 배치:

```text
RF 송수신 방식: 안테나 기반 OTA
TX/RX 안테나 거리: 약 10 cm
TX/RX 배치: 두 SDR 안테나를 옆에 나란히 세운 근거리 배치
```

따라서 실험 2 결과는 유선 직결 결과가 아니라 `10 cm OTA session-held-out` 결과입니다.

최종 결과:

```text
공식 모델: ResNet1D [I,Q]
평균 accuracy: 0.7579
최저 class recall: 0.6558
ensemble accuracy: 0.7594
```

실험 2는 여기서 완료로 둡니다. 실험 3에서는 약 `1 m` OTA 거리 조건에서 feature fusion이 성능 저하를 줄이는지 검증합니다.

## 폴더 구조

```text
config/       실험 설정 파일
data/         exp2 raw/processed 데이터
docs/         보조 문서
results/      checkpoint, metrics, reports, figures
sourcecode/   exp2 소스코드
tests/        exp2 테스트
```

## 필요 환경

예상 로컬 환경:

```text
project root: D:\ai_projects\SDR
venv: D:\ai_projects\SDR\.venv
radioconda: C:\Users\qus70\radioconda
hardware: HackRF One + RTL-SDR Blog V4
```

Python 확인:

```powershell
cd D:\ai_projects\SDR

.\.venv\Scripts\python.exe --version
```

PyTorch/CUDA 확인:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

SDR 장비 확인:

```powershell
cmd /c "call C:\Users\qus70\radioconda\Scripts\activate.bat C:\Users\qus70\radioconda && python -m src.sdr.diagnose"
```

## 기존 raw capture로 재현

아래 명령은 exp2 소스 폴더에서 실행합니다.

```powershell
cd D:\ai_projects\SDR\experiments\exp2\sourcecode
```

테스트:

```powershell
..\..\..\.venv\Scripts\python.exe -m pytest ..\tests\root_tests_snapshot -q
```

processed dataset 재생성:

```powershell
if (Test-Path ..\data\processed) { Remove-Item -Recurse -Force ..\data\processed }

..\..\..\.venv\Scripts\python.exe -m src.dataset.import_exp02_sessions --min-sessions 15
```

공식 모델 학습:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.train.run_seed_sweep --config ..\config\config.exp02.yaml --model-type resnet1d --output-root ..\results\exp02_15session_resnet
```

공식 모델 오류 분석:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.analyze_resnet_errors --config ..\config\config.exp02.yaml --data-root ..\data\processed --output-dir ..\results\reports
```

최종 보고서용 그림 재생성:

```powershell
..\..\..\.venv\Scripts\python.exe -m src.analysis.make_final_report_figures
```

전체 모델 비교, 최종 분석, 그림은 다음 문서에 있습니다.

```text
FINAL_REPORT_EXP02.md
```
