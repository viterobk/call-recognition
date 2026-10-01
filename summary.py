import json
from urllib.error import URLError
from urllib.request import Request, urlopen

MODEL_NAME = "gemma4:e2b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

PROMPT = """Сделай краткое резюме телефонного звонка на русском языке.
Напиши: кто участвовал, по какому вопросу звонили, что ответили, чем закончился разговор.
Не добавляй фактов, которых нет в транскрипции. Не добавляй названия и имена. Только суть разговора.
Ответ пришли в формате json.

Транскрипция:
{transcript}
"""


def summarize_call(transcript: str, model_name: str = MODEL_NAME) -> str:
    if not transcript.strip():
        return ""
    payload = {
        "model": model_name,
        "prompt": PROMPT.format(transcript=transcript.strip()),
        "stream": False,
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode())
    except URLError as error:
        raise RuntimeError(
            f"Не удалось вызвать {model_name}. Запустите Ollama и выполните: ollama pull {model_name}"
        ) from error
    return str(body.get("response", "")).strip()
