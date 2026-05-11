# XGBoost Environment Setup for SDR Oshea2018

이 문서는 `D:\ai_projects\SDR` 프로젝트에서 O'Shea 2018 재현 실험의 `HOS/HOM/HOC + XGBoost` baseline을 실행하기 위한 환경 설정 기록이다.

## 원칙

- 전역 Python을 사용하지 않는다.
- 반드시 프로젝트 로컬 가상환경을 사용한다.
- 패키지 설치는 항상 `python -m pip` 형식으로 실행한다.
- 사용 Python:

```text
D:\ai_projects\SDR\.venv\Scripts\python.exe
```

## 설치 완료 패키지

현재 `.venv`에 다음 패키지를 설치/확인했다.

```text
xgboost==3.2.0
scikit-learn==1.8.0
numpy==2.4.3
scipy==1.17.1
```

## 설치 명령

PowerShell에서 프로젝트 루트로 이동한다.

```powershell
cd D:\ai_projects\SDR
```

XGBoost를 프로젝트 `.venv`에 설치한다.

```powershell
D:\ai_projects\SDR\.venv\Scripts\python.exe -m pip install xgboost
```

## 설치 확인

```powershell
D:\ai_projects\SDR\.venv\Scripts\python.exe -c "import xgboost, sklearn, sys; print('python:', sys.executable); print('xgboost:', xgboost.__version__); print('sklearn:', sklearn.__version__)"
```

정상 예시:

```text
python: D:\ai_projects\SDR\.venv\Scripts\python.exe
xgboost: 3.2.0
sklearn: 1.8.0
```

## Oshea2018 HOS/XGBoost Baseline 실행

```powershell
cd D:\ai_projects\SDR\experiments\oshea2018\sourcecode
$env:PYTHONPATH = (Get-Location).Path
$PY = "D:\ai_projects\SDR\.venv\Scripts\python.exe"
$CFG = "..\config\config.oshea2018.yaml"

& $PY -m src.analysis.hos_xgboost_baseline `
  --config $CFG `
  --data-root ..\data\synthetic `
  --output-dir ..\results\synthetic_hos
```

## 역할

XGBoost는 딥러닝 모델이 아니다.

이 실험에서 XGBoost는 O'Shea 2018 논문과 동일하게 전통적 expert-feature baseline 역할을 한다.

```text
IQ window 2 x 1024
-> HOS/HOM/HOC 및 amplitude/phase/frequency 통계 feature 추출
-> XGBoost classifier
-> BASK/BFSK/BPSK 분류
```

비교 대상은 다음과 같다.

```text
HOS/XGBoost baseline
vs
raw IQ VGG-style CNN
vs
raw IQ ResNet1D
```

## Fallback 동작

`src.analysis.hos_xgboost_baseline`은 `xgboost` import가 실패하면 `sklearn.ensemble.GradientBoostingClassifier`로 fallback한다.

현재는 `xgboost==3.2.0`이 설치되어 있으므로 정상적으로 XGBoost backend가 사용되어야 한다.

결과 JSON의 `backend` 필드가 다음처럼 나오면 정상이다.

```json
{
  "backend": "xgboost"
}
```

만약 다음처럼 나오면 XGBoost가 사용되지 않은 것이다.

```json
{
  "backend": "sklearn_gradient_boosting"
}
```

이 경우 설치 확인 명령을 다시 실행한다.

## 결과 파일

실행 후 다음 파일이 생성된다.

```text
experiments\oshea2018\results\synthetic_hos\logs\hos_baseline_eval.json
experiments\oshea2018\results\synthetic_hos\hos_baseline_model.pkl
```

`hos_baseline_eval.json`에서 확인할 주요 항목:

```text
backend
accuracy
classification_report
confusion_matrix
```
