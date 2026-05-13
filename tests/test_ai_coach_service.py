import json
import sys
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.ai_coach import coach_service
from src.services.ai_coach import llm_client
from src.services.ai_coach.prompts import build_diagnosis_prompt
from src.services.ai_coach.prompts import build_question_prompt


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.headers = {}

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _sample_match_context():
    return {
        "match": {
            "provider": "statsbomb",
            "match_id": "match-1",
            "competition_name": "Mundial",
            "season_name": "2022",
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
            "selected_team": "Argentina",
            "selected_player": "Lionel Messi",
        },
        "provider_context": {
            "provider": "statsbomb",
            "capabilities": {
                "has_event_coordinates": True,
                "has_lineups": False,
                "has_team_stats": False,
                "has_player_stats": False,
                "has_xg": True,
                "has_event_timeline": True,
            },
            "limitations": [],
        },
        "tactical_metrics": {
            "total_events": 12,
            "total_passes": 8,
            "total_shots": 3,
            "field_tilt_index": 68.0,
            "directness_index": 61.0,
            "progressive_threat_index": 72.0,
            "recovery_height_index": 58.0,
            "shot_quality_index": 31.0,
            "player_influence_score": 76.0,
        },
        "tactical_summary": {
            "attacking_profile": "Equipo con tendencia vertical.",
            "territorial_profile": "Alta presencia territorial en campo rival.",
            "pressing_profile": "Recuperaciones relativamente altas.",
            "risk_profile": "Riesgo ofensivo controlado.",
            "player_profile": "Jugador con alta influencia en las acciones del partido.",
            "data_quality_note": "Contexto generado desde métricas e insights estructurados.",
        },
        "event_summary": {
            "total_canonical_events": 12,
            "event_type_counts": {"Pass": 8, "Shot": 3, "Ball Recovery": 1},
        },
        "spatial_summary": {
            "has_spatial_data": True,
            "final_third_actions_count": 4,
            "average_event_x": 64.0,
            "average_recovery_x": 61.0,
            "coordinate_system": "StatsBomb 120x80",
        },
        "insights": ["Argentina tuvo buena amenaza progresiva."],
        "suggested_questions": [
            "¿Cómo estuvo el equipo en términos generales?",
            "¿Qué debería corregir el cuerpo técnico?",
        ],
    }


def test_is_ai_coach_configured_returns_false_when_api_key_is_missing(monkeypatch):
    monkeypatch.setattr(llm_client, "_ENV_LOADED", True)
    monkeypatch.setattr(llm_client, "PROJECT_ROOT", Path("Z:/no-env-for-tests"))
    monkeypatch.delenv("AI_COACH_API_KEY", raising=False)

    assert llm_client.is_ai_coach_configured() is False


def test_get_ai_coach_config_status_does_not_expose_api_key(monkeypatch):
    monkeypatch.setattr(llm_client, "_ENV_LOADED", True)
    monkeypatch.setenv("AI_COACH_API_KEY", "super-secret-key")
    monkeypatch.setenv("AI_COACH_MODEL", "demo-model")
    monkeypatch.setenv("AI_COACH_BASE_URL", "https://example.com/v1/chat/completions")

    status = llm_client.get_ai_coach_config_status()
    serialized = json.dumps(status, ensure_ascii=False)

    assert status["configured"] is True
    assert status["model"] == "demo-model"
    assert status["api_key_configured"] is True
    assert status["model_configured"] is True
    assert status["base_url_configured"] is True
    assert "super-secret-key" not in serialized
    assert "API_COACH_API_KEY" not in serialized


def test_ai_coach_config_status_uses_defaults_without_exposing_values(monkeypatch):
    monkeypatch.setattr(llm_client, "_ENV_LOADED", True)
    monkeypatch.setattr(llm_client, "PROJECT_ROOT", Path("Z:/no-env-for-tests"))
    monkeypatch.delenv("AI_COACH_API_KEY", raising=False)
    monkeypatch.delenv("AI_COACH_MODEL", raising=False)
    monkeypatch.delenv("AI_COACH_BASE_URL", raising=False)

    status = llm_client.get_ai_coach_config_status()

    assert status["configured"] is False
    assert status["api_key_configured"] is False
    assert status["model"] == llm_client.DEFAULT_AI_COACH_MODEL
    assert status["model_configured"] is False
    assert status["base_url"] == llm_client.DEFAULT_AI_COACH_BASE_URL
    assert status["base_url_configured"] is False


def test_generate_tactical_diagnosis_returns_controlled_error_when_api_key_is_missing(monkeypatch):
    monkeypatch.setattr(
        coach_service,
        "get_ai_coach_config_status",
        lambda: {
            "configured": False,
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "message": "Falta configurar AI_COACH_API_KEY en el entorno.",
        },
    )
    monkeypatch.setattr(
        coach_service,
        "call_llm",
        lambda messages: (_ for _ in ()).throw(AssertionError("call_llm no deberia ejecutarse")),
    )

    result = coach_service.generate_tactical_diagnosis(_sample_match_context())

    assert result["ok"] is False
    assert "AI_COACH_API_KEY" in result["error"]


def test_answer_coach_question_returns_controlled_error_when_api_key_is_missing(monkeypatch):
    monkeypatch.setattr(
        coach_service,
        "get_ai_coach_config_status",
        lambda: {
            "configured": False,
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "message": "Falta configurar AI_COACH_API_KEY en el entorno.",
        },
    )
    monkeypatch.setattr(
        coach_service,
        "call_llm",
        lambda messages: (_ for _ in ()).throw(AssertionError("call_llm no deberia ejecutarse")),
    )

    result = coach_service.answer_coach_question(_sample_match_context(), "¿Cómo estuvo el equipo?")

    assert result["ok"] is False
    assert "AI_COACH_API_KEY" in result["error"]


def test_answer_coach_question_validates_empty_question():
    result = coach_service.answer_coach_question(_sample_match_context(), "   ")

    assert result["ok"] is False
    assert "vacía" in result["error"]


def test_get_suggested_questions_returns_questions_from_match_context():
    questions = coach_service.get_suggested_questions(_sample_match_context())

    assert questions == [
        "¿Cómo estuvo el equipo en términos generales?",
        "¿Qué debería corregir el cuerpo técnico?",
    ]


def test_get_suggested_questions_returns_fallback_when_missing():
    questions = coach_service.get_suggested_questions({"match": {"provider": "statsbomb"}})

    assert questions == [
        "¿Cómo estuvo el equipo en términos generales?",
        "¿Dónde generó más peligro?",
        "¿Qué debería corregir el cuerpo técnico?",
        "¿Qué jugador fue más influyente?",
        "¿Qué limitaciones tienen estos datos?",
    ]


def test_build_diagnosis_prompt_includes_required_tactical_sections():
    prompt_messages = build_diagnosis_prompt(_sample_match_context())
    combined = "\n".join(item["content"] for item in prompt_messages)

    assert "Resumen general del partido" in combined
    assert "Aspectos positivos" in combined
    assert "Puntos a mejorar" in combined
    assert "Jugadores destacados" in combined
    assert "Recomendaciones para el cuerpo técnico" in combined
    assert "Limitaciones de los datos" in combined
    assert "No menciones nombres de variables internas" in combined
    assert "dominio territorial" in combined
    assert "verticalidad del juego" in combined


def test_build_question_prompt_includes_user_question_and_match_context():
    prompt_messages = build_question_prompt(
        _sample_match_context(),
        "¿Dónde generó más peligro Argentina?",
        conversation_history=[{"role": "assistant", "content": "Contexto previo breve."}],
    )
    combined = "\n".join(item["content"] for item in prompt_messages)

    assert "¿Dónde generó más peligro Argentina?" in combined
    assert "Mundial" in combined
    assert "StatsBomb 120x80" in combined
    assert "Contexto previo breve." in combined
    assert "No menciones nombres de variables internas" in combined
    assert "amenaza ofensiva progresiva" in combined


def test_generate_tactical_diagnosis_rewrites_internal_metric_names(monkeypatch):
    monkeypatch.setattr(
        coach_service,
        "get_ai_coach_config_status",
        lambda: {
            "configured": True,
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "message": "OK",
        },
    )
    monkeypatch.setattr(
        coach_service,
        "call_llm",
        lambda messages: {
            "ok": True,
            "content": "Según el field_tilt_index y el directness_index, el equipo dominó el partido.",
            "error": "",
        },
    )

    result = coach_service.generate_tactical_diagnosis(_sample_match_context())

    assert result["ok"] is True
    assert "field_tilt_index" not in result["diagnosis"]
    assert "directness_index" not in result["diagnosis"]
    assert "dominio territorial" in result["diagnosis"]
    assert "verticalidad del juego" in result["diagnosis"]


def test_answer_coach_question_rewrites_internal_metric_names(monkeypatch):
    monkeypatch.setattr(
        coach_service,
        "get_ai_coach_config_status",
        lambda: {
            "configured": True,
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "message": "OK",
        },
    )
    monkeypatch.setattr(
        coach_service,
        "call_llm",
        lambda messages: {
            "ok": True,
            "content": "El player_influence_score de Messi fue alto y has_event_coordinates = false limita el analisis.",
            "error": "",
        },
    )

    result = coach_service.answer_coach_question(_sample_match_context(), "¿Qué jugador fue más influyente?")

    assert result["ok"] is True
    assert "player_influence_score" not in result["answer"]
    assert "influencia del jugador" in result["answer"]
    assert "coordenadas detalladas de los eventos" in result["answer"]


def test_call_llm_can_be_mocked_without_real_requests(monkeypatch):
    monkeypatch.setattr(llm_client, "_ENV_LOADED", True)
    monkeypatch.setenv("AI_COACH_API_KEY", "demo-key")
    monkeypatch.setenv("AI_COACH_MODEL", "demo-model")
    monkeypatch.setenv("AI_COACH_BASE_URL", "https://example.com/v1/chat/completions")
    monkeypatch.setattr(
        llm_client,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Diagnóstico táctico generado sin requests reales."
                        }
                    }
                ]
            }
        ),
    )

    result = llm_client.call_llm(
        [
            {"role": "system", "content": "Sistema"},
            {"role": "user", "content": "Usuario"},
        ]
    )

    assert result["ok"] is True
    assert "sin requests reales" in result["content"]


def test_call_llm_returns_friendly_message_for_quota_exceeded_http_error(monkeypatch):
    monkeypatch.setattr(llm_client, "_ENV_LOADED", True)
    monkeypatch.setenv("AI_COACH_API_KEY", "demo-key")
    monkeypatch.setenv("AI_COACH_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("AI_COACH_BASE_URL", "https://example.com/v1/chat/completions")

    class _QuotaError(HTTPError):
        def __init__(self):
            super().__init__(
                url="https://example.com/v1/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            )

        def read(self):
            return json.dumps(
                {
                    "error": {
                        "code": 429,
                        "message": "You exceeded your current quota. Quota exceeded for metric.",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                }
            ).encode("utf-8")

    def _raise_quota_error(request, timeout):
        raise _QuotaError()

    monkeypatch.setattr(llm_client, "urlopen", _raise_quota_error)

    result = llm_client.call_llm(
        [
            {"role": "system", "content": "Sistema"},
            {"role": "user", "content": "Usuario"},
        ]
    )

    assert result["ok"] is False
    assert "cuota disponible" in result["error"].lower()
    assert "billing" in result["error"].lower()


def test_call_llm_returns_friendly_message_for_service_unavailable_http_error(monkeypatch):
    monkeypatch.setattr(llm_client, "_ENV_LOADED", True)
    monkeypatch.setenv("AI_COACH_API_KEY", "demo-key")
    monkeypatch.setenv("AI_COACH_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("AI_COACH_BASE_URL", "https://example.com/v1/chat/completions")

    class _ServiceUnavailableError(HTTPError):
        def __init__(self):
            super().__init__(
                url="https://example.com/v1/chat/completions",
                code=503,
                msg="Service Unavailable",
                hdrs=None,
                fp=None,
            )

        def read(self):
            return json.dumps(
                {
                    "error": {
                        "code": 503,
                        "message": "The service is currently unavailable.",
                        "status": "UNAVAILABLE",
                    }
                }
            ).encode("utf-8")

    def _raise_service_unavailable(request, timeout):
        raise _ServiceUnavailableError()

    monkeypatch.setattr(llm_client, "urlopen", _raise_service_unavailable)

    result = llm_client.call_llm(
        [
            {"role": "system", "content": "Sistema"},
            {"role": "user", "content": "Usuario"},
        ]
    )

    assert result["ok"] is False
    assert "no está disponible" in result["error"].lower()
    assert "unos minutos" in result["error"].lower()
