# 개발 환경

## Python 패키지 매니저
- Python 환경 툴은 **uv**를 사용합니다.
- 스크립트 실행 시 `python` 대신 `uv run`을 사용하세요.
  - 예: `uv run main.py`, `uv run pytest`

## 실행 환경 분리
- **현재 코드 작성 환경**과 **VLM 학습/추론 환경**은 별개입니다.
- 모델 로드, 학습, 추론 관련 코드는 이 환경에서 직접 실행하지 마세요.
  - `Qwen2VLForConditionalGeneration`, `AutoProcessor`, `process_vision_info` 등 모델 관련 코드 실행 금지
  - 패키지 설치 확인, 문법 검사, 파일 읽기/쓰기 등 모델 무관한 작업만 이 환경에서 수행

## 추론 환경 스펙
- CUDA Version: 12.4
- Python: 3.10
