import os
import time
import logging
from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
VK_TOKEN = os.getenv("VK_TOKEN")
GROUP_ID = os.getenv("GROUP_ID")

VK_API_URL = "https://api.vk.com/method/"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# Системный промпт: Михалыч — деревенский мужик-работяга
SYSTEM_PROMPT = (
    "Ты — Михалыч, простой деревенский мужик-работяга. Тебя зовут Михалыч, "
    "и ты сам это прекрасно знаешь. Ты говоришь простым, живым, разговорным языком, "
    "без официоза и канцелярита. Можешь по-доброму пошутить, ввернуть словцо, "
    "но без грубости и мата. Ты не ассистент, не бот, не программа — ты просто Михалыч. "
    "Никогда не говори о себе как об ассистенте, ИИ или программе. "
    "Отвечай кратко и по делу, как в живом разговоре. "
    "Если к тебе обращаются по имени (Михалыч, Михалыча, Михалычу, Михалычем, Михалыче) — "
    "отвечай как Михалыч. Если речь идёт о каком-то другом Михалыче — "
    "уточни по-простому, что это ты и есть Михалыч."
)

# Общая история на всю беседу (peer_id), а не на каждого пользователя
conversations = {}


def vk_api(method, params=None):
    if params is None:
        params = {}
    params.update({"access_token": VK_TOKEN, "v": "5.199"})
    resp = requests.get(VK_API_URL + method, params=params, timeout=10)
    return resp.json()


def ask_deepseek(peer_id, message):
    if peer_id not in conversations:
        conversations[peer_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    conversations[peer_id].append({"role": "user", "content": message})
    # Держим только последние 20 сообщений + системный промпт
    history = [conversations[peer_id][0]] + conversations[peer_id][-20:]

    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": history,
                "temperature": 0.85,
                "max_tokens": 2000
            },
            timeout=30
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        conversations[peer_id].append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logging.error(f"DeepSeek error: {e}")
        return "Ох, что-то у меня в голове заклинило. Попробуй ещё разок."


def handle_message(peer_id, text):
    if text.lower() in ["/start", "/help"]:
        send_message(peer_id, "Здорово! Я Михалыч. Пиши, если чё надо.")
        return
    send_typing(peer_id)
    reply = ask_deepseek(peer_id, text)
    send_message(peer_id, reply)


def send_message(peer_id, text):
    try:
        result = vk_api("messages.send", {
            "peer_id": peer_id,
            "message": text,
            "random_id": int(time.time() * 1000)
        })
        logging.info(f"Отправка в VK: {result}")
    except Exception as e:
        logging.error(f"VK send error: {e}")


def send_typing(peer_id):
    try:
        vk_api("messages.setActivity", {
            "peer_id": peer_id,
            "type": "typing"
        })
    except:
        pass


def get_longpoll_server():
    data = vk_api("groups.getLongPollServer", {"group_id": GROUP_ID})
    logging.info(f"VK API response: {data}")
    if "error" in data:
        raise Exception(f"VK API error: {data['error']['error_msg']} (code {data['error']['error_code']})")
    return data["response"]["server"], data["response"]["key"], data["response"]["ts"]


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
                logging.info(f"Получено обновление: {update.get('type')}")
                if update.get("type") == "message_new":
                    msg = update["object"]["message"]
                    logging.info(f"Новое сообщение | peer_id={msg.get('peer_id')} | from_id={msg.get('from_id')} | text={msg.get('text')!r}")

                    # Игнорируем сообщения от самого бота
                    if msg.get("from_id", 0) < 0:
                        continue

                    text = msg.get("text", "")
                    peer_id = msg["peer_id"]

                    # В беседах (peer_id > 2000000000) реагируем на упоминание @ или на "Михалыч"
                    # В личных сообщениях реагируем на всё
                    if peer_id > 2000000000:
                        mention_pattern = f"[club{GROUP_ID}|"
                        has_mention = mention_pattern in text
                        # Простой поиск по подстроке — сработает на Михалыч, Михалыча, Михалычу и т.д.
                        has_trigger = "михалыч" in text.lower()

                        logging.info(f"DEBUG: text={text!r}, has_mention={has_mention}, has_trigger={has_trigger}")

                        if not has_mention and not has_trigger:
                            continue

                        # Убираем только упоминание @, слово "Михалыч" НЕ трогаем
                        if has_mention:
                            text = text.replace(mention_pattern, "").replace("]", "").strip()

                    if not text:
                        continue

                    handle_message(peer_id, text)

        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            logging.error(f"Poll error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
