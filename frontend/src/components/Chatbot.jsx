import { useState } from 'react';
import { useImmer } from 'use-immer';
import ChatMessages from '@/components/ChatMessages';
import ChatInput from '@/components/ChatInput';

function Chatbot() {
    /**
     * chatId: stores the current chat session id
     * messages: holds all messages in current chat, each message contains a role (user/assistant), content, loading and error property
     * newMessage: Stores the current text in the chat input (before it gets submitted)
     */
    const [chatId, setChatId] = useState(null);
    // useImmer updates state indirectly by creating a new object copy, as state updates must be performed immutably in react
    const [messages, setMessages] = useImmer([]);
    const [newMessage, setNewMessage] = useState('');


 const isLoading = messages.length && messages[messages.length - 1].loading;

  async function submitNewMessage() {
    const trimmedMessage = newMessage.trim();
    // make sure input message is not empty or that a response is loading before proceeding
    if (!trimmedMessage || isLoading) return;

    // add users message to the chat and placeholder assistant message
    setMessages(draft => [...draft,
        { role: 'user', content: trimmedMessage },
        { role: 'assistant', content: '', sources: [], loading: true }
    ]);
    setNewMessage('');

    let chatIdOrNew = chatId;
    try {
        if (!chatId) {
            // if there is no existing chat session we create a new one with the api
        const { id } = await api.createChat();
        setChatId(id);
        chatIdOrNew = id;
        }
        // use api to send user's message to the backend which returns a stream as a response
        const stream = await api.sendChatMessage(chatIdOrNew, trimmedMessage);
        /**parseSSEStream convert the SSE stream into an async iterator of text chunks
         * for each new chunk we receive we update the assistant message, creating
         * a real-time streaming effect
         */
        for await (const textChunk of parseSSEStream(stream)) {
        setMessages(draft => {
            draft[draft.length - 1].content += textChunk;
        });
        }
        //once response finishes streaming, set assistant message's loading property to false
        setMessages(draft => {
        draft[draft.length - 1].loading = false;
        });
    } catch (err) {
        // if there are any errors we set the assistant message's error property True to display an error message in the chat
        console.log(err);
        setMessages(draft => {
        draft[draft.length - 1].loading = false;
        draft[draft.length - 1].error = true;
        });
    }
  }

  return (
    // If there are no messages in chat, display the welcome message
    <div>
    
      {messages.length === 0 && (
        <div>{/* Chatbot welcome message */}</div>
      )}
      <ChatMessages
        messages={messages}
        isLoading={isLoading}
      />
      <ChatInput
        newMessage={newMessage}
        isLoading={isLoading}
        setNewMessage={setNewMessage}
        submitNewMessage={submitNewMessage}
      />
    </div>
  );
}

export default Chatbot;