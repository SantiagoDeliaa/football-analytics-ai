from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_AI_COACH_MODEL = "gpt-4o-mini"
DEFAULT_AI_COACH_BASE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 20.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_LOADED = False


def _load_local_env_file(force_reload: bool = False) -> None:
    global _ENV_LOADED
    if _ENV_LOADED and not force_reload:
        return

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            # Si el .env falla, el servicio sigue con el entorno disponible.
            pass

    _ENV_LOADED = True


def _get_ai_coach_api_key() -> str | None:
    api_key = str(os.getenv("AI_COACH_API_KEY", "") or "").strip()
    if not api_key:
        _load_local_env_file()
        api_key = str(os.getenv("AI_COACH_API_KEY", "") or "").strip()
    if not api_key:
        _load_local_env_file(force_reload=True)
        api_key = str(os.getenv("AI_COACH_API_KEY", "") or "").strip()
    return api_key or None


def _normalize_base_url(base_url: str | None) -> str:
    resolved = str(base_url or DEFAULT_AI_COACH_BASE_URL).strip() or DEFAULT_AI_COACH_BASE_URL
    parsed = urlparse(resolved)
    if not parsed.scheme or not parsed.netloc:
        return DEFAULT_AI_COACH_BASE_URL
    if resolved.rstrip("/").endswith("/v1"):
        return f"{resolved.rstrip('/')}/chat/completions"
    return resolved


def _extract_content(parsed_response: dict[str, Any]) -> str:
    choices = parsed_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0] or {}
    message = first_choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    text = first_choice.get("text")
    if isinstance(text, str):
        return text.strip()

    return ""


def _sanitize_provider_detail(detail: str) -> str:
    normalized = str(detail or "").strip()
    if not normalized:
        return ""
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = normalized.replace("`", "")
    return normalized


def _friendly_ai_coach_error_message(
    error_text: str,
    *,
    status_code: int | None = None,
) -> str:
    normalized = _sanitize_provider_detail(error_text).lower()

    if status_code == 429 or "quota exceeded" in normalized or "resource_exhausted" in normalized:
        return (
            "El AI Coach no tiene cuota disponible en el provider configurado. "
            "Revisá billing, plan o límites del proyecto e intentá nuevamente más tarde."
        )
    if status_code == 401 or "unauthorized" in normalized or "invalid api key" in normalized:
        return "La credencial configurada para el AI Coach no es válida o no tiene permisos."
    if status_code == 403 or "forbidden" in normalized or "permission denied" in normalized:
        return "El provider rechazó el acceso del AI Coach. Revisá permisos, proyecto y modelo configurado."
    if "model" in normalized and "not found" in normalized:
        return "El modelo configurado para el AI Coach no está disponible en el provider actual."

    return ""


def is_ai_coach_configured() -> bool:
    return _get_ai_coach_api_key() is not None


def get_ai_coach_config_status() -> dict[str, Any]:
    api_key = _get_ai_coach_api_key()
    model = str(os.getenv("AI_COACH_MODEL", DEFAULT_AI_COACH_MODEL) or DEFAULT_AI_COACH_MODEL).strip()
    base_url = _normalize_base_url(os.getenv("AI_COACH_BASE_URL"))
    model_from_env = bool(str(os.getenv("AI_COACH_MODEL", "") or "").strip())
    base_url_from_env = bool(str(os.getenv("AI_COACH_BASE_URL", "") or "").strip())

    if api_key:
        return {
            "configured": True,
            "model": model,
            "base_url": base_url,
            "api_key_configured": True,
            "model_configured": model_from_env,
            "base_url_configured": base_url_from_env,
            "message": "AI Tactical Coach configurado correctamente.",
        }

    return {
        "configured": False,
        "model": model,
        "base_url": base_url,
        "api_key_configured": False,
        "model_configured": model_from_env,
        "base_url_configured": base_url_from_env,
        "message": "Falta configurar AI_COACH_API_KEY en el entorno.",
    }


def call_llm(
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> dict[str, Any]:
    config = get_ai_coach_config_status()
    if not config["configured"]:
        return {
            "ok": False,
            "content": "",
            "error": str(config["message"]),
        }

    if not isinstance(messages, list) or not messages:
        return {
            "ok": False,
            "content": "",
            "error": "No se pudo llamar al AI Coach porque faltan mensajes para el modelo.",
        }

    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    request = Request(
        config["base_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_get_ai_coach_api_key()}",
            "User-Agent": "football-analytics-ai/1.0",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            raw_payload = response.read().decode("utf-8")
        parsed = json.loads(raw_payload)
        provider_error = parsed.get("error")
        if provider_error:
            if isinstance(provider_error, dict):
                error_message = str(provider_error.get("message") or "El provider devolvió un error.")
            else:
                error_message = str(provider_error)
            friendly_message = _friendly_ai_coach_error_message(error_message)
            return {
                "ok": False,
                "content": "",
                "error": friendly_message or f"Error del provider del AI Coach: {error_message}",
            }

        content = _extract_content(parsed)
        if not content:
            return {
                "ok": False,
                "content": "",
                "error": "El provider del AI Coach devolvió una respuesta vacía.",
            }

        return {
            "ok": True,
            "content": content,
            "error": "",
        }
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8").strip()
        except Exception:
            detail = ""
        friendly_message = _friendly_ai_coach_error_message(detail or str(exc), status_code=exc.code)
        if friendly_message:
            return {"ok": False, "content": "", "error": friendly_message}
        message = f"Error HTTP llamando al AI Coach: {exc}"
        if detail:
            message = f"{message}. Detalle: {detail}"
        return {"ok": False, "content": "", "error": message}
    except URLError as exc:
        return {
            "ok": False,
            "content": "",
            "error": f"Error de red llamando al AI Coach: {exc}",
        }
    except TimeoutError as exc:
        return {
            "ok": False,
            "content": "",
            "error": f"Timeout llamando al AI Coach: {exc}",
        }
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "content": "",
            "error": f"Respuesta JSON inválida desde el AI Coach: {exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "content": "",
            "error": f"Fallo inesperado llamando al AI Coach: {exc}",
        }
