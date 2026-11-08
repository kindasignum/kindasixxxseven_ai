"""
Шлюз к нейросети (по умолчанию Groq). Работает на сервере в Нидерландах и принимает запросы от VK-бота
по HTTPS. Ключ провайдера хранится только здесь. Внешних библиотек не нужно — только стандартный Python.

Переменные окружения:
    GATEWAY_TOKEN   — длинная случайная строка; бот присылает её в заголовке Authorization: Bearer ... (обязательно)
    GROQ_API_KEY    — ключ Groq (обязательно; можно назвать PROVIDER_KEY — для другого провайдера)
    PROVIDER_URL    — адрес chat/completions (по умолчанию Groq: https://api.groq.com/openai/v1/chat/completions)
    ALLOWED_MODELS  — разрешённые модели через запятую (по умолчанию llama-3.3-70b-versatile,llama-3.1-8b-instant)
    PORT            — порт (по умолчанию 3000)
    MAX_TOKENS_CAP  — потолок max_tokens в одном ответе (по умолчанию 300)
    MAX_INPUT_CHARS — потолок размера запроса в символах (по умолчанию 12000)
    DAILY_LIMIT     — запросов в сутки (по умолчанию 1000), после — ответ 429
    MINUTE_LIMIT    — запросов в минуту (по умолчанию 20)

Запросы:
    GET  /health    — проверка, что шлюз жив (без токена, ничего секретного не показывает)
    POST /v1/chat   — тело: {"model", "system", "messages": [{"role","content"}], "max_tokens"}
                      ответ: {"text": "...", "usage": {"input": N, "output": N}}
Тексты сообщений в лог не пишутся — только модель, размеры и расход токенов.
"""
import hmac
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")

API_KEY = (os.getenv("GROQ_API_KEY") or os.getenv("PROVIDER_KEY") or "").strip()
TOKEN = os.getenv("GATEWAY_TOKEN", "").strip()
PROVIDER_URL = os.getenv("PROVIDER_URL", "https://api.groq.com/openai/v1/chat/completions").strip()
PORT = int(os.getenv("PORT", "3000"))
ALLOWED_MODELS = {m.strip() for m in os.getenv(
    "ALLOWED_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",") if m.strip()}
MAX_TOKENS_CAP = int(os.getenv("MAX_TOKENS_CAP", "300"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "1000"))
MINUTE_LIMIT = int(os.getenv("MINUTE_LIMIT", "20"))
MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGES = 60
UPSTREAM_TIMEOUT = 30

_lock = threading.Lock()
_day = {"date": None, "count": 0}
_minute = []  # время последних запросов (скользящее окно 60 секунд)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)  # некоторые модели присылают «рассуждения» — убираем


def take_quota() -> str:
    """Занимает место в лимитах. Пустая строка — можно, иначе — причина отказа."""
    now = time.time()
    today = datetime.now(timezone.utc).date().isoformat()
    with _lock:
        if _day["date"] != today:
            _day.update(date=today, count=0)
        _minute[:] = [x for x in _minute if now - x < 60]
        if _day["count"] >= DAILY_LIMIT:
            return "суточный лимит запросов исчерпан"
        if len(_minute) >= MINUTE_LIMIT:
            return "слишком много запросов в минуту"
        _day["count"] += 1
        _minute.append(now)
    return ""


def validate(data: dict):
    """Проверяет тело запроса. Возвращает (тело для провайдера, None) или (None, текст ошибки)."""
    model = data.get("model")
    if model not in ALLOWED_MODELS:
        return None, f"модель не разрешена; доступны: {', '.join(sorted(ALLOWED_MODELS))}"
    msgs = data.get("messages")
    if not isinstance(msgs, list) or not msgs or len(msgs) > MAX_MESSAGES:
        return None, f"messages: список от 1 до {MAX_MESSAGES} сообщений"
    total, clean = 0, []
    for m in msgs:
        if (not isinstance(m, dict) or m.get("role") not in ("user", "assistant")
                or not isinstance(m.get("content"), str) or not m["content"].strip()):
            return None, "каждое сообщение: {role: user|assistant, content: непустая строка}"
        total += len(m["content"])
        clean.append({"role": m["role"], "content": m["content"]})
    if clean[0]["role"] != "user" or clean[-1]["role"] != "user":
        return None, "первое и последнее сообщение должны быть от user"
    system = data.get("system", "")
    if not isinstance(system, str):
        return None, "system должен быть строкой"
    total += len(system)
    if total > MAX_INPUT_CHARS:
        return None, f"запрос слишком большой ({total} > {MAX_INPUT_CHARS} символов)"
    try:
        max_tokens = int(data.get("max_tokens", 150))
    except (TypeError, ValueError):
        return None, "max_tokens должен быть числом"
    max_tokens = max(1, min(max_tokens, MAX_TOKENS_CAP))
    body = {"model": model, "max_tokens": max_tokens,
            "messages": ([{"role": "system", "content": system}] if system.strip() else []) + clean}
    temp = data.get("temperature")
    if isinstance(temp, (int, float)) and 0 <= temp <= 2:
        body["temperature"] = float(temp)
    return body, None


def call_provider(body: dict):
    """Запрос к провайдеру. Возвращает (код, ответ-словарь). Один повтор при сбое сети или 5xx."""
    raw = json.dumps(body).encode("utf-8")
    last = (502, {"error": "provider_unreachable"})
    for attempt in range(2):
        req = urllib.request.Request(PROVIDER_URL, data=raw, method="POST", headers={
            "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
            "User-Agent": "llm-gateway/1.0"})  # без своего User-Agent защита провайдера может отклонить запрос
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            retry_after = e.headers.get("retry-after")
            last = (e.code, {"error": "provider_error", "upstream_status": e.code, "detail": detail,
                             "retry_after": retry_after})
            if e.code < 500:
                return last
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = (502, {"error": "provider_unreachable", "detail": str(e)[:300]})
        except ValueError:
            return 502, {"error": "provider_bad_json"}
        if attempt == 0:
            time.sleep(1)
    return last


class Handler(BaseHTTPRequestHandler):
    server_version = "gateway"

    def log_message(self, fmt, *args):  # свой лог вместо стандартного: пишем только метод и путь (без адресов и токенов)
        log.info("Запрос: %s", (self.requestline or "").split(" HTTP")[0][:120])

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/health"):  # «/» нужен и для проверок живости на стороне хостинга
            return self._send(200, {"ok": True, "provider_key_set": bool(API_KEY), "token_set": bool(TOKEN),
                                    "models": sorted(ALLOWED_MODELS)})
        self._send(404, {"error": "not found"})

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/chat":
            return self._send(404, {"error": "not found"})
        if not TOKEN or not API_KEY:
            return self._send(500, {"error": "шлюз не настроен: нет GATEWAY_TOKEN или GROQ_API_KEY"})
        auth = self.headers.get("Authorization", "")
        given = auth[7:] if auth.startswith("Bearer ") else ""
        if not hmac.compare_digest(given.encode("utf-8"), TOKEN.encode("utf-8")):
            log.warning("Отказ: неверный токен")
            return self._send(401, {"error": "неверный токен"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return self._send(400, {"error": "пустой или слишком большой запрос"})
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            return self._send(400, {"error": "тело запроса — не JSON-объект"})
        body, err = validate(data)
        if err:
            return self._send(400, {"error": err})
        why = take_quota()
        if why:
            log.warning("Отказ по лимиту: %s", why)
            return self._send(429, {"error": why})
        started = time.time()
        try:
            code, resp = call_provider(body)
        except Exception as e:  # noqa: BLE001
            log.exception("Неожиданная ошибка")
            return self._send(500, {"error": "internal", "detail": type(e).__name__})
        if code != 200:
            log.error("Провайдер вернул %s: %s", code, str(resp.get("detail", resp))[:200])
            # 429 от провайдера отдаём как 429, остальное — как 502, чтобы бот отличал «лимит» от «сломалось»
            return self._send(429 if code == 429 else 502, resp)
        try:
            text = THINK_RE.sub("", resp["choices"][0]["message"]["content"] or "").strip()
            u = resp.get("usage") or {}
            usage = {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)}
        except (KeyError, IndexError, TypeError):
            log.error("Неожиданный формат ответа провайдера")
            return self._send(502, {"error": "provider_bad_format"})
        log.info("OK модель=%s вход=%s выход=%s за %.1f с", body["model"], usage["input"], usage["output"],
                 time.time() - started)
        self._send(200, {"text": text, "usage": usage})


def main():
    if not API_KEY:
        log.warning("GROQ_API_KEY не задан — запросы к нейросети работать не будут")
    if not TOKEN:
        log.warning("GATEWAY_TOKEN не задан — шлюз будет отвечать ошибкой на все запросы")
    elif len(TOKEN) < 24:
        log.warning("GATEWAY_TOKEN слишком короткий — возьмите случайную строку от 32 символов")
    log.info("Шлюз запущен на порту %s; провайдер: %s; модели: %s; лимиты: %s/сутки, %s/мин, max_tokens ≤ %s",
             PORT, PROVIDER_URL, ", ".join(sorted(ALLOWED_MODELS)), DAILY_LIMIT, MINUTE_LIMIT, MAX_TOKENS_CAP)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()