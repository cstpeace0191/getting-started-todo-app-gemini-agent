"""classifier 단위 테스트 — 실제 모델은 호출하지 않는다 (ADK 의 Runner 를 mock)"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.ai import agent, classifier
from src.models import Classification

AI_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE", "GOOGLE_API_KEY", "GOOGLE_API_KEY_FILE",
    "GOOGLE_GENAI_USE_VERTEXAI", "AI_MODEL", "AI_ENABLED", "AI_CLASSIFY_ENABLED",
)


def _event(text=None, *, final=True, error_code=None, error_message=None):
    parts = [SimpleNamespace(text=text)] if text is not None else []
    return SimpleNamespace(
        error_code=error_code,
        error_message=error_message,
        content=SimpleNamespace(parts=parts) if parts else None,
        is_final_response=lambda: final,
    )


def _json_event(payload: dict):
    return _event(json.dumps(payload))


def fake_runner(*events, captured=None, error=None):
    """agent.Runner 를 대체하는 가짜. run_async 가 미리 정한 이벤트를 흘려 준다."""

    class FakeRunner:
        def __init__(self, *, app_name, agent, session_service):
            if captured is not None:
                captured["agent"] = agent

        async def run_async(self, *, user_id, session_id, new_message):
            if captured is not None:
                captured["prompt"] = new_message.parts[0].text
            if error:
                raise error
            for e in events:
                yield e

        async def close(self):
            pass

    return FakeRunner


@pytest.fixture(autouse=True)
def gemini_key(monkeypatch):
    for n in AI_ENV:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


async def _classify(*args, runner=None, **kw):
    with patch.object(agent, "Runner", runner):
        return await classifier.classify_item(*args, **kw)


async def test_it_returns_parsed_classification():
    cap = {}
    out = {"category": "study", "priority": "high", "due_date": None, "due_time": None}
    result = await _classify("내일까지 클라우드 과제 제출",
                             runner=fake_runner(_json_event(out), captured=cap))

    assert result == Classification(category="study", priority="high")
    a = cap["agent"]
    assert a.model == "gemini-3.8-flash"
    assert a.tools == []                          # 도구 없음
    assert a.output_schema is Classification
    assert "<todo>내일까지 클라우드 과제 제출</todo>" in cap["prompt"]


async def test_it_ignores_non_final_events():
    out = {"category": "work", "priority": "low"}
    runner = fake_runner(_event("생각 중…", final=False), _json_event(out))
    result = await _classify("보고서", runner=runner)
    assert result.category == "work"


async def test_it_is_disabled_without_any_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    cap = {}
    assert await _classify("장보기", runner=fake_runner(captured=cap)) is None
    assert cap == {}
    assert classifier.is_enabled() is False


@pytest.mark.parametrize("name", ["AI_ENABLED", "AI_CLASSIFY_ENABLED"])
def test_it_can_be_disabled_by_env(monkeypatch, name):
    monkeypatch.setenv(name, "false")
    assert classifier.is_enabled() is False


def test_it_reads_google_key_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY")
    key_file = tmp_path / "google_api_key"
    key_file.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_API_KEY_FILE", str(key_file))

    assert agent.read_google_api_key() == "file-key"
    assert classifier.is_enabled() is True


def test_it_is_enabled_with_vertex_ai(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert classifier.is_enabled() is True


def test_it_falls_back_to_claude_with_anthropic_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    assert agent.resolve_model() == "claude-opus-5-5"
    a = agent.build_agent(name="t", instruction="x", schema=Classification)
    assert a.model.model == "anthropic/claude-opus-5-5"  # LiteLlm 으로 연결


def test_explicit_claude_model_needs_anthropic_key(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "claude-sonnet-5-5")   # Google 키만 있는 상태
    assert agent.resolve_model() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    assert agent.resolve_model() == "claude-sonnet-5-5"


async def test_it_returns_none_on_runner_error():
    assert await _classify("회의 준비", runner=fake_runner(error=RuntimeError("boom"))) is None


async def test_it_returns_none_on_error_event():
    e = _event(final=True, error_code="429", error_message="quota")
    assert await _classify("무언가", runner=fake_runner(e)) is None


async def test_it_returns_none_without_final_text():
    assert await _classify("무언가", runner=fake_runner(_event(final=True))) is None


async def test_it_returns_none_on_schema_mismatch():
    bad = _json_event({"category": "nonsense", "priority": "high"})
    assert await _classify("무언가", runner=fake_runner(bad)) is None


async def test_it_returns_none_on_non_json_text():
    assert await _classify("무언가", runner=fake_runner(_event("분류 불가"))) is None


async def test_it_sends_today_and_returns_due_date():
    cap = {}
    out = {"category": "work", "priority": "high", "due_date": "2026-11-26", "due_time": "09:00"}
    result = await _classify("내일 오전 9시까지 보고서", "2026-11-25",
                             runner=fake_runner(_json_event(out), captured=cap))

    assert "오늘 날짜: 2026-11-25" in cap["prompt"]
    assert (result.due_date, result.due_time) == ("2026-11-26", "09:00")


async def test_it_does_not_send_today_when_not_given():
    cap = {}
    out = {"category": "work", "priority": "high"}
    await _classify("보고서", runner=fake_runner(_json_event(out), captured=cap))
    assert "오늘 날짜" not in cap["prompt"]


@pytest.mark.parametrize(
    "due_date,due_time,expected",
    [
        ("2026-02-31", "09:00", (None, None)),        # 존재하지 않는 날짜 → 일정 버림
        ("tomorrow", "09:00", (None, None)),          # 형식 오류
        ("2026-11-26", "9am", ("2026-11-26", None)),  # 시간만 잘못 → 시간만 버림
        (None, "09:00", (None, None)),                # 날짜 없이 시간만 → 버림
    ],
)
async def test_it_discards_invalid_ai_due_values(due_date, due_time, expected):
    out = {"category": "work", "priority": "low", "due_date": due_date, "due_time": due_time}
    result = await _classify("x", "2026-11-25", runner=fake_runner(_json_event(out)))
    assert (result.due_date, result.due_time) == expected
