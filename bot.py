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

conversations = {}


def vk_api(method, params=None):
    if params is None:
        params = {}
    params.update({"access_token": VK_TOKEN, "v": "5.199"})
    resp = requests.get(VK_API_URL + method, params=params, timeout=10)
    return resp.json()


def ask_deepseek(user_id, message):
    if user_id not in conversations:
        conversations[user_id] = [
            {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко и по делу."}
        ]
    conversations[user_id].append({"role": "user", "content": message})
    history = conversations[user_id][-20:]

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
                "temperature": 0.7,
                "max_tokens": 2000
            },
            timeout=30
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        conversations[user_id].append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logging.error(f"DeepSeek error: {e}")
        return "Извини, произошла ошибка. Попробуй позже."


def handle_message(peer_id, text, from_id):
    if text.lower() in ["/start", "/help"]:
        send_message(peer_id, "Привет! Я бот на DeepSeek. Упомяни меня через @ и напиши сообщение.")
        return
    send_typing(peer_id)
    reply = ask_deepseek(from_id, text)
    send_message(peer_id, reply)


def send_message(peer_id, text):
    try:
        vk_api("messages.send", {
            "peer_id": peer_id,
            "message": text,
            "random_id": int(time.time() * 1000)
        })
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
                if update.get("type") == "message_new":
                    msg = update["object"]["message"]

                    # Игнорируем сообщения от самого бота
                    if msg.get("from_id", 0) < 0:
                        continue

                    text = msg.get("text", "")
                    peer_id = msg["peer_id"]
                    from_id = msg.get("from_id")

                    # В беседах (peer_id > 2000000000) реагируем только на упоминание
                    # В личных сообщениях реагируем на всё
                    if peer_id > 2000000000:
                        mention_pattern = f"[club{GROUP_ID}|"
                        if mention_pattern not in text:
                            continue  # Нет упоминания — пропускаем
                        # Убираем упоминание из текста
                        text = text.replace(mention_pattern, "").replace("]", "").strip()

                    if not text:
                        continue  # Пустое сообщение — пропускаем

                    handle_message(peer_id, text, from_id)

        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            logging.error(f"Poll error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
