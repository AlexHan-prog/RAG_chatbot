import json
from time import time

async def create_chat(rdb, chat_id: str, title: str = "New Chat"):
    created_at = int(time())
    await rdb.hset(
        f"chat:{chat_id}",
        mapping={
            "id": chat_id,
            "title": title,
            "created_at": created_at,
        }
    )

async def chat_exists(rdb, chat_id: str) -> bool:
    return await rdb.exists(f"chat:{chat_id}") == 1

async def get_chat_metadata(rdb, chat_id: str) -> dict | None:
    """
    returns:
        metadata for a particular chatId
    """
    data = await rdb.hgetall(f"chat:{chat_id}")
    if not data:
        return None
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "created_at": int(data.get("created_at", 0)),
    }

async def list_chats(rdb) -> list[dict]:
    """
    Retrieves all chat sessions for the sidebar listing
    """
    keys = await rdb.keys("chat:*")
    chats = []

    for key in keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        # ignore messages lists, because we store:
        # chat:ID and chat:ID:messages
        if key_str.endswith(":messages"):
            continue

        data = await rdb.hgetall(key_str)
        if data:
            chats.append({
                "id": data.get("id"),
                "title": data.get("title"),
                "created_at": int(data.get("created_at", 0)),
            })

    # sort so newest chats appear first
    chats.sort(key=lambda c: c["created_at"], reverse=True)
    return chats

async def get_messages(rdb, chat_id: str) -> list[dict]:

    # retrieves messages for a particular chatId
    # range is 0 to -1 meaning retrieve the entire list ##
   
    raw_messages = await rdb.lrange(f"chat:{chat_id}:messages", 0, -1)
    messages = []

    for msg in raw_messages:
        if isinstance(msg, bytes):
            msg = msg.decode()
        # convert json to objects e.g: 
        # [
        #    {"role": "user", "content": "Hello"},
        #    {"role": "assistant", "content": "Hi!"}
        # ]#
        messages.append(json.loads(msg))

    return messages

async def append_message(rdb, chat_id: str, role: str, content: str):
    """
    Adds a message to the conversation history
    """
    message = json.dumps({"role": role, "content": content})
    await rdb.rpush(f"chat:{chat_id}:messages", message)

async def update_chat_title(rdb, chat_id: str, title: str):
    await rdb.hset(f"chat:{chat_id}", "title", title)

async def delete_chat(rdb, chat_id: str):
    """Remove a chat and its messages from Redis."""
    await rdb.delete(f"chat:{chat_id}", f"chat:{chat_id}:messages")