import json
from time import time

async def create_chat(rdb, chat_id: str, title: str = "New Chat"):
    """
    Create a new chat session with basic metadata.

    This function:
    - Generates a Unix timestamp for when the chat was created
    - Stores the chat's id, title and created_at in a Redis hash
    - Uses the key pattern ``chat:{chat_id}`` for the metadata

    Args:
        rdb:
            Async Redis client/connection supporting ``hset``.
        chat_id (str):
            Unique identifier for the chat session.
        title (str, optional):
            Human-friendly chat title shown in the UI. Defaults to "New Chat".
    """
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
    """Check whether a chat with the given id exists.

    This function:
    - Looks up the Redis key ``chat:{chat_id}``
    - Returns ``True`` if the hash exists, otherwise ``False``

    Args:
        rdb:
            Async Redis client/connection supporting ``exists``.
        chat_id (str):
            Chat identifier to check.

    Returns:
        bool: ``True`` if the chat metadata hash exists, ``False`` otherwise.
    """
    return await rdb.exists(f"chat:{chat_id}") == 1

async def get_chat_metadata(rdb, chat_id: str) -> dict | None:
    """Retrieve metadata for a single chat.

    This function:
    - Reads the redis hash at ``chat:{chat_id}``
    - returns a normalised dictionry of fields if present
    - returns ``None`` if the chat does not exist

    Args:
        rdb:
            async Redis client/connection supporting ``hgetall``.
        chat_id (str):
            identifier of the chat whose metadata is requested.

    Returns:
        dict | None:
            - dict: chat metadata with keys ``id``, ``title`` and ``created_at``.
            - None: if no data is stored for the given ``chat_id``.
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
    Retrieve all chat sessions.

    This function:
    - Scans Redis for keys matching ``chat:*``
    - ignores message lists with the ``:messages`` suffix
    - loads metadata for each chat hash
    - normalizes and sorts chats so the newest appear first

    Args:
        rdb:
            redis client supporting ``keys`` and ``hgetall``.

    Returns:
        list[dict]:
            List of chat metadata dictionaries with keys ``id``, ``title`` and
            ``created_at``, ordered by ``created_at`` descending.
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
    """Retrieve all messages for a given chat.

    This function:
    - Reads the Redis list at ``chat:{chat_id}:messages``
    - Decodes any byte strings to text
    - Parses each list element from JSON into a Python dict

    The stored JSON objects typically look like:

    .. code-block:: json

        {"role": "user", "content": "Hello"}

    Args:
        rdb:
            redis client/connection supporting ``lrange``.
        chat_id (str):
            Chat identifier whose message history should be returned.

    Returns:
        list[dict]:
            Ordered list of message objects for the chat.
    """
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
    """Append a new message to a chat's history.

    This function:
    - Serializes the message (role + content) as JSON
    - Appends it to the Redis list ``chat:{chat_id}:messages``

    Args:
        rdb:
            Async Redis client/connection supporting ``rpush``.
        chat_id (str):
            Identifier of the chat to append the message to.
        role (str):
            Message role, e.g. ``"user"`` or ``"assistant"``.
        content (str):
            Raw message text content.
    """
    message = json.dumps({"role": role, "content": content})
    await rdb.rpush(f"chat:{chat_id}:messages", message)

async def update_chat_title(rdb, chat_id: str, title: str):
    """Update the title of an existing chat.

    This function:
    - Overwrites the ``title`` field in the Redis hash ``chat:{chat_id}``.

    Args:
        rdb:
            Async Redis client/connection supporting ``hset``.
        chat_id (str):
            Identifier of the chat whose title should be updated.
        title (str):
            New human-readable title for the chat.
    """
    await rdb.hset(f"chat:{chat_id}", "title", title)

async def delete_chat(rdb, chat_id: str):
    """Remove a chat and all of its messages from Redis.

    This function:
    - Deletes the chat metadata hash ``chat:{chat_id}``
    - Deletes the associated messages list ``chat:{chat_id}:messages``

    Args:
        rdb:
            Async Redis client/connection supporting ``delete``.
        chat_id (str):
            Identifier of the chat to remove.
    """
    await rdb.delete(f"chat:{chat_id}", f"chat:{chat_id}:messages")