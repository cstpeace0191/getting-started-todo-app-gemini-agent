"""Google ADK(Agent Development Kit) 공통 실행 모듈

classifier(도구 없는 단일 분류)와 planner(도구를 쓰는 일정 브리핑)가 함께 쓴다.
각 기능은 ADK 의 LlmAgent 로 정의하고, 이 모듈의 run_structured() 가
Runner 로 에이전트를 실행해 output_schema(Pydantic) 로 검증된 결과를 돌려준다.

모델 선택 (AI_MODEL 환경변수, 생략 시 자동):
  - gemini-*  : ADK 기본 Gemini 연동. 인증은 아래 중 하나
      · GOOGLE_API_KEY / GOOGLE_API_KEY_FILE (Docker/K8s Secret 마운트용 파일 경로)
      · GOOGLE_GENAI_USE_VERTEXAI=true (Vertex AI, GKE 의 Workload Identity/ADC 사용)
  - claude-*  : LiteLLM 을 통해 Anthropic Claude 사용
      · ANTHROPIC_API_KEY / ANTHROPIC_API_KEY_FILE
  AI_MODEL 이 없으면 Google 인증이 있을 때 Gemini, 아니면 Anthropic 키가 있을 때 Claude 를 쓴다.
사용할 수 있는 인증이 없거나 AI_ENABLED=false (구버전 이름 AI_CLASSIFY_ENABLED 도 인정) 이면
AI 기능은 꺼진다.
"""
import logging
import os
import uuid
from typing import Any, Callable, TypeVar

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

APP_NAME = "todo_ai"
USER_ID = "todo-user"

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

T = TypeVar("T", bound=BaseModel)


def _read_secret(key_env: str, file_env: str) -> str | None:
    """키 파일(file_env)이 있으면 그 내용을, 없으면 환경변수(key_env)를 돌려준다."""
    file_path = os.getenv(file_env)
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    return os.getenv(key_env) or None


def read_api_key() -> str | None:
    """Anthropic API 키 (Claude 모델 사용 시)"""
    return _read_secret("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE")


def read_google_api_key() -> str | None:
    """Google(Gemini) API 키"""
    return _read_secret("GOOGLE_API_KEY", "GOOGLE_API_KEY_FILE")


def _use_vertex() -> bool:
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")


def _has_google_auth() -> bool:
    return _use_vertex() or read_google_api_key() is not None


def _is_claude(model: str) -> bool:
    return model.lower().startswith(("claude", "anthropic/"))


def resolve_model() -> str | None:
    """사용할 모델 이름. 쓸 수 있는 인증이 없으면 None."""
    explicit = os.getenv("AI_MODEL")
    if explicit:
        has_auth = read_api_key() is not None if _is_claude(explicit) else _has_google_auth()
        return explicit if has_auth else None
    if _has_google_auth():
        return DEFAULT_GEMINI_MODEL
    if read_api_key() is not None:
        return DEFAULT_CLAUDE_MODEL
    return None


def is_enabled() -> bool:
    for name in ("AI_ENABLED", "AI_CLASSIFY_ENABLED"):
        if os.getenv(name, "true").lower() in ("false", "0", "no"):
            return False
    return resolve_model() is not None


def _build_model(model: str) -> Any:
    """ADK 에 넘길 모델 객체(Gemini 는 문자열, Claude 는 LiteLlm)."""
    if _is_claude(model):
        from google.adk.models.lite_llm import LiteLlm  # litellm 은 Claude 사용 시에만 로드

        name = model if model.startswith("anthropic/") else f"anthropic/{model}"
        return LiteLlm(model=name, api_key=read_api_key())

    # Gemini 클라이언트는 환경변수에서 키를 읽는다. 키 파일로 받은 경우 환경변수로 옮겨 둔다.
    if not _use_vertex() and not os.getenv("GOOGLE_API_KEY"):
        key = read_google_api_key()
        if key:
            os.environ["GOOGLE_API_KEY"] = key
    return model


def build_agent(
    *,
    name: str,
    instruction: str,
    schema: type[BaseModel],
    tools: list[Callable[..., Any]] | None = None,
) -> LlmAgent | None:
    """앱 전용 ADK 에이전트. 결과는 schema 로 강제되고, 도구는 넘겨 준 함수만 쓸 수 있다."""
    model = resolve_model()
    if model is None:
        return None
    return LlmAgent(
        name=name,
        model=_build_model(model),
        # instruction 의 {…} 는 ADK 가 세션 상태 변수로 해석하므로 중괄호를 이스케이프한다
        instruction=instruction.replace("{", "{{").replace("}", "}}"),
        tools=tools or [],
        output_schema=schema,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )


async def run_structured(agent: LlmAgent, prompt: str, schema: type[T]) -> T | None:
    """에이전트를 실행해 구조화된 결과를 schema 로 검증해 돌려준다. 실패 시 None.

    할 일 저장(CRUD)은 AI 가 실패해도 항상 동작해야 하므로 예외를 밖으로 던지지 않는다.
    호출마다 일회용 세션을 만들어, 요청 간 대화 내용이 섞이거나 메모리에 쌓이지 않게 한다.
    """
    session_service = InMemorySessionService()
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)
    session_id = uuid.uuid4().hex
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

    final_text: str | None = None
    try:
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=message
        ):
            if event.error_code or event.error_message:
                logger.warning(
                    "ADK 에이전트 오류(%s): %s", event.error_code, event.error_message
                )
                return None
            if event.is_final_response() and event.content and event.content.parts:
                text = "".join(p.text for p in event.content.parts if p.text)
                if text:
                    final_text = text
    except Exception as e:  # noqa: BLE001 - 모델/네트워크/인증 오류도 CRUD 를 막으면 안 된다
        logger.warning("ADK 에이전트 실행 실패: %s", e)
        return None
    finally:
        await runner.close()

    if not final_text:
        logger.warning("ADK 에이전트가 결과를 돌려주지 않았습니다.")
        return None
    try:
        return schema.model_validate_json(final_text)
    except ValidationError as e:
        logger.warning("AI 결과가 스키마와 맞지 않습니다: %s", e)
        return None
