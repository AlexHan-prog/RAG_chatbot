const BASE_URL = import.meta.env.VITE_API_URL;

export async function createChat() {
  const res = await fetch(`${BASE_URL}/chats`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Failed to create chat');
  return data;
}

export async function deleteChat(chatId) {
  const res = await fetch(`${BASE_URL}/chats/${chatId}`, {
    method: 'DELETE',
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Failed to delete chat');
  return data;
}

export async function getChats() {
  let res;

  try {
    res = await fetch(`${BASE_URL}/chats`);
  } catch (err) {
    throw new Error("Could not connect to backend");
  }
  
  let data = null;
  try {
    data = await res.json();
  } catch {}

  if (!res.ok) {
    throw new Error(data?.detail || 'Failed to load chats');
  }

  return data;
}

export async function getChatMessages(chatId) {
  const res = await fetch(`${BASE_URL}/chats/${chatId}/messages`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Failed to load messages');
  return data;
}

export async function sendChatMessage(chatId, message) {
  const res = await fetch(`${BASE_URL}/chats/${chatId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Request failed');
  return data;
}