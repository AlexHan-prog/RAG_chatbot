import { useState } from 'react';
import { useImmer } from 'use-immer';
import ChatMessages from '@/components/ChatMessages';
import ChatInput from '@/components/ChatInput';

function Chatbot() {
    /**
     * chatId: stores the current chat session id
     * messages:
     */
    const [chatId, setChatId] = useState(null);
    const [messages, setMessages] = useImmer([]);
    const [newMessage, setNewMessage] = useState('');


 const isLoading = messages.length && messages[messages.length - 1].loading;

  async function submitNewMessage() {
    // Implemented in the next section
  }

  return (
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