import os
import time
import logging
import json
from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
VK_TOKEN = os.getenv("VK_TOKEN")
GROUP_ID = os.getenv("GROUP_ID")

VK_API_URL = "https://api.vk.com/method/"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

SYSTEM_PROMPT = (
    "Ты — Михалыч, простой деревенский мужик-работяга. Тебя зовут Михалыч, "
    "и ты сам это прекрасно знаешь. Ты говоришь простым, живым, разговорным языком, "
    "без официоза и канцелярита. Можешь по-доброму пошутить, ввернуть словцо, "
    "но без грубости и мата. Ты не ассистент, не бот, не программа — ты просто Михалыч. "
    "Никогда не говори о себе как об ассистенте, ИИ или программе. "
    "Отвечай кратко и по делу, как в живом разговоре. "
    "Если к тебе обращаются по имени (Михалыч, Михалыча, Михалычу, Михалычем, Михалыче) — "
    "отвечай как Михалыч. Если речь идёт о каком-то другом Михалыче — "
    "уточни по-простому, что это ты и есть Михалыч. "
    "ВАЖНО: если пользователь спрашивает о погоде, ты ОБЯЗАН вызвать функцию get_weather, "
    "чтобы получить актуальные данные. Не выдумывай погоду сам, всегда используй функцию."
)

conversations = {}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Получить актуальную погоду в указанном городе. Вызывай эту функцию всегда, когда речь идёт о погоде.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города, например, 'Новокуйбышевск' или 'Москва'",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def vk_api(method, params=None, retries=2):
    if params is None:
        params = {}
    params.update({"access_token": VK_TOKEN, "v": "5.199"})
    for attempt in range(retries):
        try:
            resp = requests.get(VK_API_URL + method, params=params, timeout=15)
            return resp.json()
        except Exception as e:
            logging.error(f"VK API error (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return {"error": {"error_msg": str(e)}}
            time.sleep(2)


def get_weather(city):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        description = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind_speed = current["windspeedKmph"]
        return (
            f"Погода в {city}: {description}, температура {temp_c}°C "
            f"(ощущается как {feels_like}°C), влажность {humidity}%, "
            f"ветер {wind_speed} км/ч."
        )
    except Exception as e:
        logging.error(f"Ошибка получения погоды: {e}")
        return f"Не удалось получить погоду для города {city}."


def call_deepseek(payload, retries=2):
    """Вызов DeepSeek с ретраями и увеличенным таймаутом."""
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_error = e
            logging.error(f"DeepSeek error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    raise last_error


def ask_deepseek(peer_id, message):
    if peer_id not in conversations:
        conversations[peer_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    conversations[peer_id].append({"role": "user", "content": message})
    history = [conversations[peer_id][0]] + conversations[peer_id][-20:]

    # Если в сообщении есть слово "погод" — принудительно вызываем get_weather
    tool_choice = "auto"
    if "погод" in message.lower():
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        logging.info("Обнаружено слово 'погод' — форсируем вызов get_weather")

    try:
        response_data = call_deepseek({
            "model": "deepseek-chat",
            "messages": history,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": 0.85,
            "max_tokens": 2000
        })
        message_obj = response_data["choices"][0]["message"]

        if message_obj.get("tool_calls"):
            tool_call = message_obj["tool_calls"][0]
            if tool_call["function"]["name"] == "get_weather":
                args = json.loads(tool_call["function"]["arguments"])
                city = args.get("city")
                logging.info(f"DeepSeek запросил погоду для города: {city}")
                weather_result = get_weather(city)
                logging.info(f"Погода получена: {weather_result}")

                conversations[peer_id].append(message_obj)
                conversations[peer_id].append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": weather_result
                })

                final_data = call_deepseek({
                    "model": "deepseek-chat",
                    "messages": conversations[peer_id],
                    "temperature": 0.85,
                    "max_tokens": 2000
                })
                final_reply = final_data["choices"][0]["message"]["content"]
                conversations[peer_id].append({"role": "assistant", "content": final_reply})
                return final_reply

        reply = message_obj.get("content") or "Ох, что-то я задумался..."
        conversations[peer_id].append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        logging.error(f"ask_deepseek failed: {e}")
        return "Ох, что-то у меня в голове заклинило. Попробуй ещё разок."


def handle_message(peer_id, text):
    if text.lower() in ["/start", "/help"]:
        send_message(peer_id, "Здорово! Я Михалыч. Пиши, если чё надо.")
        return
    send_typing(peer_id)
    reply = ask_deepseek(peer_id, text)
    send_message(peer_id, reply)


def send_message(peer_id, text):
    result = vk_api("messages.send", {
        "peer_id": peer_id,
        "message": text,
        "random_id": int(time.time() * 1000)
    })
    logging.info(f"Отправка в VK: {result}")


def send_typing(peer_id):
    vk_api("messages.setActivity", {
        "peer_id": peer_id,
        "type": "typing"
    })


def get_longpoll_server():
    data = vk_api("groups.getLongPollServer", {"group_id": GROUP_ID})
    logging.info(f"VK API response: {data}")
    if "error" in data:
        raise Exception(f"VK API error: {data['error']['error_msg']} (code {data['error']['error_code']})")
    return data["response"]["server"], data["response"]["key"], data["response"]["ts"]


def extract_message_from_update(update):
    """Извлекает сообщение из объекта обновления VK. Работает и для message_new, и для message_reply."""
    obj = update.get("object", {})
    if isinstance(obj, dict):
        if "message" in obj and isinstance(obj["message"], dict):
            return obj["message"]
        if "peer_id" in obj:
            return obj
    return None


def process_message(msg, is_reply_event=False):
    """Обрабатывает входящее сообщение."""
    from_id = msg.get("from_id", 0)
    if from_id < 0:
        # Сообщение от самого сообщества (бота)
        return

    text = msg.get("text", "")
    peer_id = msg.get("peer_id")

    if peer_id > 2000000000:
        mention_pattern = f"[club{GROUP_ID}|"
        has_mention = mention_pattern in text
        has_trigger = "михалыч" in text.lower()

        reply_msg = msg.get("reply_message")
        is_reply_to_bot = bool(reply_msg) and reply_msg.get("from_id", 0) < 0

        logging.info(
            f"DEBUG: has_mention={has_mention}, has_trigger={has_trigger}, "
            f"is_reply_to_bot={is_reply_to_bot}, is_reply_event={is_reply_event}"
        )

        # Для message_reply: обрабатываем только если это ответ на сообщение бота
        if is_reply_event and not is_reply_to_bot:
            logging.info("Пропущено: это reply не к сообщению бота")
            return

        if not has_mention and not has_trigger and not is_reply_to_bot:
            return

        if has_mention:
            text = text.replace(mention_pattern, "").replace("]", "").strip()

    if not text:
        return

    handle_message(peer_id, text)


def main():
    if not all([DEEPSEEK_API_KEY, VK_TOKEN, GROUP_ID]):
        print("Ошибка: проверь .env файл — нужны DEEPSEEK_API_KEY, VK_TOKEN, GROUP_ID")
        return

    logging.info("Запуск бота через Long Poll API...")
    server, key, ts = get_longpoll_server()
    logging.info(f"Long Poll server: {server}")

    while True:
        try:
            resp = requests.get(server, params={
                "act": "a_check",
                "key": key,
                "ts": ts,
                "wait": 25
            }, timeout=30)
            data = resp.json()

            if "failed" in data:
                if data["failed"] in (1, 2, 3):
                    server, key, ts = get_longpoll_server()
                continue

            ts = data["ts"]

            for update in data.get("updates", []):
                update_type = update.get("type")
                logging.info(f"Получено обновление: {update_type}")

                if update_type == "message_new":
                    msg = extract_message_from_update(update)
                    if msg is None:
                        logging.warning(f"Не удалось извлечь сообщение из update: {update}")
                        continue
                    logging.info(
                        f"Новое сообщение | peer_id={msg.get('peer_id')} | "
                        f"from_id={msg.get('from_id')} | text={msg.get('text')!r}"
                    )
                    process_message(msg, is_reply_event=False)

                elif update_type == "message_reply":
                    msg = extract_message_from_update(update)
                    if msg is None:
                        logging.warning(f"Не удалось извлечь сообщение из message_reply: {update}")
                        continue
                    logging.info(
                        f"Reply | peer_id={msg.get('peer_id')} | "
                        f"from_id={msg.get('from_id')} | text={msg.get('text')!r}"
                    )
                    process_message(msg, is_reply_event=True)

        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            logging.error(f"Poll error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
