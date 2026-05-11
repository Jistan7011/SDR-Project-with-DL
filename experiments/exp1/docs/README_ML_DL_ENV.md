# AI / ML / DL 개발환경 참고 문서

이 문서는 Codex, Claude Code, Cursor, VS Code 등 AI 코딩 도구가 내 Windows 머신러닝/딥러닝 개발환경을 이해하고, 새 AI 프로젝트를 만들 때 따라야 할 기준을 정리한 문서이다.

목적:

- 현재 PC의 Python / CUDA / PyTorch 개발환경 정리
- 새 머신러닝/딥러닝 프로젝트 생성 절차 표준화
- 가상환경 실행 / 종료 / 패키지 설치 방식 정리
- Codex / Claude Code가 전역 Python이 아닌 프로젝트별 `.venv`를 사용하도록 지시
- PyTorch CUDA 버전 설치 명령어를 명확히 기록

---

## 1. 현재 PC 기본 세팅

### OS / GPU

```text
OS: Windows
GPU: NVIDIA GeForce RTX 4060
AI project root: D:\ai_projects
```

### CUDA Toolkit

현재 시스템 CUDA Toolkit은 11.8이다.

```text
CUDA Toolkit: 11.8
nvcc version: 11.8.89
```

확인 명령:

```powershell
nvcc --version
```

현재 확인된 CUDA Toolkit 출력:

```text
Cuda compilation tools, release 11.8, V11.8.89
```

### cuDNN

현재 cuDNN은 CUDA Toolkit 11.8 설치 경로의 다음 폴더에 수동 복사해 둔 상태이다.

```text
CUDA\bin
CUDA\include
CUDA\lib
```

주의:

- PyTorch pip CUDA wheel은 보통 필요한 CUDA runtime 라이브러리를 자체적으로 포함하거나 의존 패키지로 관리한다.
- 따라서 PyTorch만 사용할 때는 시스템 cuDNN 수동 설치가 직접적으로 필요하지 않을 수 있다.
- 하지만 CUDA C/C++ 직접 개발, custom CUDA extension, 일부 라이브러리 빌드, cuDNN 직접 참조 코드에서는 현재 CUDA/cuDNN 설치가 의미가 있다.

---

## 2. Python 구성 원칙

현재 PC에는 전역 Python 3.14.3이 설치되어 있다.

```text
Global Python: Python 3.14.3
```

하지만 머신러닝/딥러닝 프로젝트에서는 전역 Python 3.14.3을 사용하지 않는다.

딥러닝/머신러닝 프로젝트는 다음 기준을 따른다.

```text
Python version for AI projects: Python 3.11.x
Virtual environment: project-local .venv
Package install method: python -m pip
```

현재 검증된 기준 프로젝트:

```text
Project: torch_test
Path: D:\ai_projects\torch_test
Virtual environment: D:\ai_projects\torch_test\.venv
Python: 3.11.9
PyTorch: 2.7.1+cu118
CUDA runtime used by PyTorch: 11.8
GPU available in PyTorch: True
```

---

## 3. 절대 지켜야 할 원칙

### 원칙 1. 전역 Python에 패키지를 설치하지 않는다

하지 말 것:

```powershell
pip install torch
pip install numpy
pip install opencv-python
```

이 방식은 현재 어떤 Python에 설치되는지 헷갈릴 수 있다.

반드시 프로젝트 가상환경을 활성화한 뒤 다음 형식을 사용한다.

```powershell
python -m pip install 패키지명
```

---

### 원칙 2. 프로젝트마다 `.venv`를 따로 만든다

예시:

```text
D:\ai_projects\torch_test\.venv
D:\ai_projects\yolo_test\.venv
D:\ai_projects\sdr_ai\.venv
D:\ai_projects\capstone_ai\.venv
```

이유:

- 프로젝트마다 필요한 패키지 버전이 다를 수 있다.
- PyTorch, TensorFlow, OpenCV, YOLO, SDR 관련 패키지는 버전 충돌이 자주 난다.
- 전역 Python 하나에 모든 패키지를 넣으면 나중에 환경이 쉽게 망가진다.

---

### 원칙 3. VS Code / Codex / Claude Code는 반드시 `.venv` Python을 사용해야 한다

올바른 Python 예시:

```text
D:\ai_projects\torch_test\.venv\Scripts\python.exe
```

잘못된 Python 예시:

```text
C:\Users\qus70\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

AI 코딩 도구가 Python을 실행해야 할 때는 항상 현재 프로젝트의 `.venv\Scripts\python.exe`를 기준으로 실행해야 한다.

---

## 4. 기존 AI 프로젝트 실행 방법

예: `D:\ai_projects\torch_test`

### PowerShell에서 프로젝트로 이동

```powershell
cd D:\ai_projects\torch_test
```

### 가상환경 실행

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

정상적으로 실행되면 프롬프트 앞에 `(.venv)`가 붙는다.

```powershell
(.venv) PS D:\ai_projects\torch_test>
```

### 가상환경 종료

```powershell
deactivate
```

---

## 5. 현재 Python / pip / torch 확인 명령

가상환경을 켠 상태에서 다음을 확인한다.

```powershell
where.exe python
python --version
python -m pip --version
python -m pip show torch
```

정상 예시:

```text
D:\ai_projects\torch_test\.venv\Scripts\python.exe
Python 3.11.9
torch 2.7.1+cu118
```

---

## 6. PyTorch CUDA 동작 확인 명령

가상환경을 켠 상태에서 다음 명령으로 PyTorch가 GPU를 사용할 수 있는지 확인한다.

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda runtime:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

현재 내 PC에서 기대하는 정상 출력 형태:

```text
torch: 2.7.1+cu118
cuda runtime: 11.8
cuda available: True
gpu: NVIDIA GeForce RTX 4060
```

중요:

- `cuda available: True`가 나와야 GPU 사용 가능 상태이다.
- `torch.__version__`에 `+cu118`이 붙어야 CUDA 11.8용 PyTorch이다.
- `torch.version.cuda`가 `11.8`이면 PyTorch가 CUDA 11.8 runtime을 사용한다는 뜻이다.

---

## 7. 새 AI 프로젝트 생성 표준 절차

예: 새 프로젝트 이름이 `my_ai_project`일 때

### 1단계: 프로젝트 폴더 생성

```powershell
D:
cd D:\ai_projects
mkdir my_ai_project
cd my_ai_project
```

### 2단계: Python 3.11 가상환경 생성

```powershell
py -3.11 -m venv .venv
```

### 3단계: 가상환경 실행

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 4단계: pip 기본 도구 업데이트

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 5단계: PyTorch CUDA 11.8 설치

현재 내 PC 기준 권장 PyTorch 설치 명령:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6단계: CUDA 확인

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

정상 예시:

```text
2.7.1+cu118
11.8
True
NVIDIA GeForce RTX 4060
```

---

## 8. 자주 쓰는 패키지 설치 명령

### 기본 머신러닝 / 데이터 분석

```powershell
python -m pip install numpy pandas matplotlib scikit-learn jupyter ipykernel tqdm
```

### OpenCV

```powershell
python -m pip install opencv-python
```

### PyTorch 기반 딥러닝

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### YOLO / 객체 탐지

```powershell
python -m pip install ultralytics opencv-python
```

### Flask / 웹 서버 / 대시보드

```powershell
python -m pip install flask flask-socketio requests
```

### SDR + AI 실험 기본 패키지

```powershell
python -m pip install numpy scipy matplotlib scikit-learn tqdm
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 9. requirements.txt 관리

### 현재 가상환경 패키지 저장

```powershell
python -m pip freeze > requirements.txt
```

### 저장된 패키지 확인

```powershell
type requirements.txt
```

### 다른 환경에서 복구

```powershell
python -m pip install -r requirements.txt
```

주의:

- PyTorch CUDA 버전은 설치 인덱스 URL이 중요하다.
- `requirements.txt`만으로 CUDA wheel이 제대로 복구되지 않을 수 있다.
- 새 환경에서는 PyTorch를 다음 명령으로 명시 설치하는 방식을 우선한다.

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

그 후 나머지 패키지를 설치한다.

---

## 10. VS Code 설정

### 인터프리터 선택

VS Code에서 다음 메뉴를 사용한다.

```text
Ctrl + Shift + P
Python: Select Interpreter
```

선택해야 하는 인터프리터:

```text
D:\ai_projects\현재프로젝트\.venv\Scripts\python.exe
```

예시:

```text
D:\ai_projects\torch_test\.venv\Scripts\python.exe
```

### 프로젝트 고정 설정

프로젝트 폴더에 다음 파일을 만든다.

```text
.vscode\settings.json
```

예시 내용:

```json
{
  "python.defaultInterpreterPath": "D:\\ai_projects\\torch_test\\.venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true
}
```

새 프로젝트에서는 `torch_test` 부분을 현재 프로젝트 이름으로 바꾼다.

---

## 11. CPU용 PyTorch가 잘못 잡혔을 때

잘못된 상태 예시:

```text
torch: 2.x.x+cpu
cuda runtime: None
cuda available: False
```

이 경우 원인은 대부분 다음 중 하나이다.

1. VS Code가 전역 Python 3.14를 사용 중이다.
2. 현재 PowerShell에서 `.venv`가 활성화되지 않았다.
3. 현재 가상환경에 CPU용 PyTorch가 설치되어 있다.
4. `pip`가 잘못된 Python에 연결되어 있다.

점검 순서:

```powershell
where.exe python
python --version
python -m pip show torch
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

현재 프로젝트의 `.venv`가 아닌 Python이 잡히면 VS Code interpreter 또는 PowerShell 가상환경 실행 상태를 수정한다.

---

## 12. PyTorch 재설치 방법

현재 가상환경에 CPU용 PyTorch가 잘못 설치되었거나 CUDA 인식이 안 될 때:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

재확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## 13. CUDA 버전 개념 정리

CUDA 관련 버전은 세 종류가 있다.

| 구분 | 확인 명령 | 의미 |
|---|---|---|
| NVIDIA Driver CUDA | `nvidia-smi` | 드라이버가 지원 가능한 최대 CUDA 버전 |
| CUDA Toolkit | `nvcc --version` | 시스템에 설치된 CUDA 개발 도구 |
| PyTorch CUDA runtime | `torch.version.cuda` | PyTorch wheel이 실제 사용하는 CUDA runtime |

중요:

- `nvidia-smi`에 표시되는 CUDA 버전과 `torch.version.cuda`는 달라도 된다.
- PyTorch 코드가 실제 사용하는 CUDA runtime은 `torch.version.cuda`로 확인한다.
- 현재 내 기준 환경은 PyTorch CUDA runtime 11.8이다.

---

## 14. 프로젝트 구조 권장안

```text
D:\ai_projects
├─ torch_test
│  ├─ .venv
│  ├─ requirements.txt
│  └─ README_ML_DL_ENV.md
│
├─ yolo_test
│  ├─ .venv
│  ├─ train.py
│  ├─ predict.py
│  ├─ requirements.txt
│  └─ datasets
│
├─ sdr_ai
│  ├─ .venv
│  ├─ capture_iq.py
│  ├─ train.py
│  ├─ requirements.txt
│  └─ data
│
└─ capstone_ai
   ├─ .venv
   ├─ app.py
   ├─ model
   ├─ requirements.txt
   └─ README_env.md
```

---

## 15. Codex / Claude Code에게 전달할 핵심 지시문

AI 코딩 도구가 이 프로젝트를 다룰 때는 다음 원칙을 따른다.

```text
이 프로젝트는 Windows 환경에서 실행된다.
전역 Python 3.14는 사용하지 않는다.
AI/ML/DL 프로젝트는 Python 3.11 기반 프로젝트별 .venv를 사용한다.
현재 프로젝트의 Python은 반드시 .venv\Scripts\python.exe를 사용한다.
패키지 설치는 pip 단독 명령이 아니라 python -m pip 형식을 사용한다.
PyTorch는 CUDA 11.8 wheel을 사용한다.
PyTorch 설치 명령은 다음과 같다:

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

CUDA 확인은 다음 명령으로 한다:

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

새 의존성을 추가하면 requirements.txt를 갱신한다:

python -m pip freeze > requirements.txt

가상환경 실행:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

가상환경 종료:

deactivate
```

---

## 16. 최종 운영 원칙

```text
1. 전역 Python은 건드리지 않는다.
2. 새 AI 프로젝트마다 Python 3.11 기반 .venv를 만든다.
3. VS Code / Codex / Claude Code는 반드시 .venv Python을 사용한다.
4. PyTorch는 CUDA 11.8용 wheel을 명시적으로 설치한다.
5. 패키지 설치는 항상 python -m pip 형식을 사용한다.
6. 설치 후 torch.cuda.is_available()를 확인한다.
7. requirements.txt를 프로젝트별로 관리한다.
8. 환경이 꼬이면 Python 경로 -> torch 위치 -> CUDA 사용 가능 여부 순서로 점검한다.
```
