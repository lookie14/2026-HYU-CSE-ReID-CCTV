# Intelligent CCTV ReID

카메라 프레임에서 사람을 찾아(YOLO) 특징 벡터를 뽑고(BoT ReID + PCA 128차원), 벡터 DB(pgvector)에서 같은 사람을 찾아 ID를 붙이는 시스템.

```
frontend/index.html ──WebSocket──▶ backend (Spring Boot :8080) ──RabbitMQ──▶ ai-server (YOLO + ReID)
                                       │  ▲                                        │
frontend/dashboard.html ◀──WebSocket───┘  └──────────── RabbitMQ (결과) ◀───────────┘
                                       │
                              PostgreSQL+pgvector, Redis (docker compose)
```

| 폴더 | 내용 |
|------|------|
| `frontend/` | `index.html` 카메라 프레임 전송, `dashboard.html` 결과 대시보드 |
| `backend/` | Spring Boot 4 (Java 17+). 프레임 수신 → 큐 발행, 결과 수신 → 인물 매칭(임계치 0.35) → 저장 |
| `ai-server/` | FastAPI + RabbitMQ 소비자. YOLO26n 사람 탐지 → BoT(ResNet50) 2048차원 → PCA 128차원 |

## 1. 체크포인트

ReID 가중치 `ai-server/checkpoints/bot_joint_kr100_last.pth`(약 100MB)는 용량 문제로 레포에 포함하지 않았다. 실행하려면 같은 방식으로 학습한 BoT 가중치를 이 경로에 두어야 한다(`state_dict` 키를 가진 PyTorch 체크포인트, 구조는 `reid_core/models/bot.py`). PCA 파일(`pca_128_j_kr100.npz`)은 레포에 포함돼 있다.

YOLO 가중치(`yolo26n.pt`)는 ultralytics가 첫 실행 때 자동으로 받는다.

모델: ResNet50 기반 BoT를 Market-1501(751명) + 한국 CCTV(AI-Hub 한국인 재식별 데이터, 100명)로 처음부터 함께 학습(60 epoch, 마지막 체크포인트). PCA는 이 모델 특징으로 다시 fit했다.

## 2. 접속 정보 (.env)

비밀번호 등 접속 정보는 레포에 없고 환경 변수로 넣는다. AI 서버와 백엔드는 같은 RabbitMQ(예: CloudAMQP)를 쓴다.

```bash
cp .env.example .env   # 값 채우기
set -a; source .env; set +a
```

`ai-server/config.yaml`은 `${RABBITMQ_*}`를, `backend/.../application.yml`과 `docker-compose.yaml`은 `${RABBITMQ_*}`·`${POSTGRES_*}`를 읽는다. 포트·SSL은 각 설정 파일에서 바꾼다(AI 서버 5671/SSL, 백엔드 5672).

## 3. 실행

**DB·Redis**
```bash
cd backend && docker compose up -d     # pgvector(5433), redis(6379)
```

**백엔드** (Java 17+)
```bash
cd backend && ./gradlew bootRun        # :8080, /ws/stream, /ws/dashboard
```
> 시작 시 `DummyDataLoader`가 DB가 비어 있으면 성능 측정용 랜덤 벡터 1만 건을 넣는다. 필요 없으면 이 클래스의 `@Component`를 지운다.

**AI 서버** (Python 3.11, GPU 권장)
```bash
cd ai-server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # CUDA torch는 pytorch.org 명령으로 먼저 설치
uvicorn main:app --host 0.0.0.0 --port 8000
```
- `GET /health` 로드된 모델 확인
- `POST /embed` (multipart `file`) 이미지 한 장 → 사람별 bbox와 128차원 벡터. RabbitMQ 없이 모델만 확인할 때 쓴다.

**프런트엔드**
- `frontend/index.html`을 브라우저로 열고 시작 → 카메라 프레임이 `ws://localhost:8080/ws/stream`으로 전송된다. 다른 기기(휴대폰)에서 쓰려면 ngrok 등으로 터널을 열고 파일 안 `targetUrl`을 `wss://...` 주소로 바꾼다.
- `frontend/dashboard.html`을 열면 매칭 결과가 실시간으로 표시된다.

## 설정 요약

| 항목 | 위치 | 값 |
|------|------|----|
| ReID 가중치 / PCA | `ai-server/config.yaml` | `checkpoints/bot_joint_kr100_last.pth`, `checkpoints/pca_128_j_kr100.npz` |
| 배치 | `ai-server/config.yaml` `reid.batch` | 최대 16프레임 / 30ms |
| 결과 메시지 | `ai-server/config.yaml` `publish_grouped` | `true` = 프레임당 1건(schemaVersion 2) |
| 매칭 임계치 | `backend/.../service/ReidService.java` | 코사인 거리 0.35 이하면 같은 사람 |

모델을 바꾸면 PCA와 임계치도 같이 바꾸고, 기존 DB 벡터와는 호환되지 않으므로 DB를 비운다.
