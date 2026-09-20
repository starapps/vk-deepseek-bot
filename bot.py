import os
import time
import logging
import json
import re
from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
VK_TOKEN = os.getenv("VK_TOKEN")
GROUP_ID = os.getenv("GROUP_ID")

VK_API_URL = "https://api.vk.com/method/"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

MAX_HISTORY = 50

# Слова-триггеры, на которые бот реагирует без @
TRIGGER_WORDS = [
    "михалыч",
    "тракторист",
    "колхозник",
    "деревенщина",
    "шеф",
    "босс",
    "командир",
]

SYSTEM_PROMPT = (
    "Ты — Михалыч, простой деревенский мужик-работяга. Тебя зовут Михалыч, "
    "и ты сам это прекрасно знаешь. Ты говоришь простым, живым, разговорным языком, "
    "без официоза и канцелярита. Можешь по-доброму пошутить, ввернуть словцо, "
    "но без грубости и мата. Ты не ассистент, не бот, не программа — ты просто Михалыч. "
    "Никогда не говори о себе как об ассистенте, ИИ или программе. "
    "Отвечай кратко и по делу, как в живом разговоре. "
    "В беседе несколько человек — ты видишь, кто что пишет, в формате 'Имя: сообщение'. "
    "Обращайся к тому, кто к тебе обратился по имени. "
    "Отвечай просто текстом, без префикса 'Михалыч:'. "
    "ВАЖНО: если пользователь спрашивает о погоде, ты ОБЯЗАН вызвать функцию get_weather, "
    "чтобы получить актуальные данные."
)

conversations = {}
user_names_cache = {}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Получить актуальную погоду в указанном городе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города, например, 'Новокуйбышевск'",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def vk_api(method, params=None, retries=2, timeout=10):
    if params is None:
        params = {}
    params.update({"access_token": VK_TOKEN, "v": "5.199"})
    for attempt in range(retries):
        try:
            resp = requests.get(VK_API_URL + method, params=params, timeout=timeout)
            return resp.json()
        except Exception as e:
            logging.error(f"VK API error (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return {"error": {"error_msg": str(e)}}
            time.sleep(1)


def get_user_name(user_id):
    """Возвращает имя пользователя. С таймаутом 5 секунд и кэшем."""
    if user_id < 0:
        return "Михалыч"
    if user_id in user_names_cache:
        return user_names_cache[user_id]
    try:
        data = vk_api("users.get", {"user_ids": user_id}, timeout=5, retries=1)
        if "response" in data and data["response"]:
            user = data["response"][0]
            first = user.get("first_name", "").strip()
            last = user.get("last_name", "").strip()
            name = (first + " " + last).strip() or f"user_{user_id}"
            user_names_cache[user_id] = name
            return name
    except Exception as e:
        logging.error(f"Не удалось получить имя для {user_id}: {e}")
    fallback = f"user_{user_id}"
    user_names_cache[user_id] = fallback
    return fallback


def add_to_history(peer_id, role, content):
    if peer_id not in conversations:
        conversations[peer_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    conversations[peer_id].append({"role": role, "content": content})
    if len(conversations[peer_id]) > MAX_HISTORY + 1:
        conversations[peer_id] = [conversations[peer_id][0]] + conversations[peer_id][-MAX_HISTORY:]


def get_weather(city):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        return (
            f"Погода в {city}: {current['weatherDesc'][0]['value']}, "
            f"температура {current['temp_C']}°C "
            f"(ощущается как {current['FeelsLikeC']}°C), "
            f"влажность {current['humidity']}%, "
            f"ветер {current['windspeedKmph']} км/ч."
        )
    except Exception as e:
        logging.error(f"Ошибка получения погоды: {e}")
        return f"Не удалось получить погоду для города {city}."


def call_deepseek(payload, retries=2):
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
    messages_for_deepseek = list(conversations.get(peer_id, []))

    tool_choice = "auto"
    if "погод" in message.lower():
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        logging.info("Обнаружено слово 'погод' — форсируем вызов get_weather")

    try:
        response_data = call_deepseek({
            "model": "deepseek-chat",
            "messages": messages_for_deepseek,
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

                extended = messages_for_deepseek + [
                    message_obj,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": weather_result
                    }
                ]

                final_data = call_deepseek({
                    "model": "deepseek-chat",
                    "messages": extended,
                    "temperature": 0.85,
                    "max_tokens": 2000
                })
                return final_data["choices"][0]["message"]["content"]

        return message_obj.get("content") or "Ох, что-то я задумался..."

    except Exception as e:
        logging.error(f"ask_deepseek failed: {e}")
        return "Ох, что-то у меня в голове заклинило. Попробуй ещё разок."


def send_message(peer_id, text):
    result = vk_api("messages.send", {
        "peer_id": peer_id,
        "message": text,
        "random_id": int(time.time() * 1000)
    })
    logging.info(f"Отправка в VK: {result}")


def send_typing(peer_id):
    vk_api("messages.setActivity", {"peer_id": peer_id, "type": "typing"})


def handle_message(peer_id, text):
    if text.lower() in ["/start", "/help"]:
        reply = "Здорово! Я Михалыч. Пиши, если чё надо."
        add_to_history(peer_id, "assistant", reply)
        send_message(peer_id, reply)
        return

    send_typing(peer_id)
    reply = ask_deepseek(peer_id, text)
    add_to_history(peer_id, "assistant", reply)
    send_message(peer_id, reply)


def get_longpoll_server():
    data = vk_api("groups.getLongPollServer", {"group_id": GROUP_ID})
    logging.info(f"VK API response: {data}")
    if "error" in data:
        raise Exception(f"VK API error: {data['error']['error_msg']} (code {data['error']['error_code']})")
    return data["response"]["server"], data["response"]["key"], data["response"]["ts"]


def process_message(msg):
    """Обрабатывает message_new. Проверяет триггеры и решает, отвечать ли."""
    from_id = msg.get("from_id", 0)
    if from_id < 0:
        return

    text = msg.get("text", "")
    peer_id = msg.get("peer_id")

    mention_pattern = f"[club{GROUP_ID}|"
    if mention_pattern in text:
        text_clean = re.sub(rf"\[club{GROUP_ID}\|[^\]]*\]", "", text).strip()
    else:
        text_clean = text

    # Сохраняем в историю (с именем автора)
    author = get_user_name(from_id)
    if text_clean:
        add_to_history(peer_id, "user", f"{author}: {text_clean}")

    if peer_id > 2000000000:
        has_mention = mention_pattern in text
        has_trigger = any(w in text.lower() for w in TRIGGER_WORDS)
        reply_msg = msg.get("reply_message")
        is_reply_to_bot = bool(reply_msg) and reply_msg.get("from_id", 0) < 0

        logging.info(
            f"DEBUG: has_mention={has_mention}, has_trigger={has_trigger}, "
            f"is_reply_to_bot={is_reply_to_bot}"
        )

        if not has_mention and not has_trigger and not is_reply_to_bot:
            return

        if not text_clean:
            return

        handle_message(peer_id, text_clean)
    else:
        if text_clean:
            handle_message(peer_id, text_clean)


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

                # Обрабатываем ТОЛЬКО message_new. VK дублирует ответы в message_reply — игнорируем.
                if update_type == "message_new":
                    obj = update.get("object", {})
                    msg = obj.get("message", obj)
                    logging.info(
                        f"message_new | peer_id={msg.get('peer_id')} | "
                        f"from_id={msg.get('from_id')} | text={msg.get('text')!r}"
                    )
                    process_message(msg)

        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            logging.error(f"Poll error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
