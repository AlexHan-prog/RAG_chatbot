import { useEffect, useRef, useState } from 'react';
import useAutosize from '@/hooks/useAutosize';
import sendIcon from '@/assets/images/send.svg';

function ChatInput({ 
  newMessage,
  isLoading,
  setNewMessage,
  submitNewMessage,
  selectedMode,
  setSelectedMode 
}) {
  const textareaRef = useAutosize(newMessage);
  const [showModeMenu, setShowModeMenu] = useState(false);
  const modeMenuRef = useRef(null);

  const modes = [
    { value: 'auto', label: 'Auto', short: 'Auto (default)' },
    { value: 'llm', label: 'LLM', short: 'LLM' },
    { value: 'rag', label: 'RAG', short: 'RAG' },
    { value: 'mcp', label: 'MCP', short: 'MCP' },
  ];

  const currentMode = modes.find(mode => mode.value === selectedMode) || modes[0];

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      submitNewMessage();
    }
  }

  useEffect(() => {
    function handleClickOutside(e) {
      if (modeMenuRef.current && !modeMenuRef.current.contains(e.target)) {
        setShowModeMenu(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className='sticky bottom-0 shrink-0 bg-white py-4'>
      <div className='flex items-center gap-2'>
        <div className='relative shrink-0' ref={modeMenuRef}>
          <button
            type='button'
            disabled={isLoading}
            onClick={() => setShowModeMenu(prev => !prev)}
            title={`Current mode: ${currentMode.label}`}
            className={`
              flex h-[44px] min-w-[44px] items-center justify-center rounded-2xl border px-3
              text-sm font-semibold shadow-sm
              transition-all duration-200 ease-out
              hover:scale-[1.03] hover:shadow-md
              active:scale-95
              disabled:cursor-not-allowed disabled:opacity-50
              ${selectedMode === 'auto'
                ? 'border-primary-blue/20 bg-white text-main-text hover:bg-primary-blue/10'
                : 'border-primary-blue/30 bg-primary-blue/10 text-primary-blue hover:bg-primary-blue/15'
              }
            `}
          >
            <span
              key={currentMode.value}
              className='inline-block animate-[fadeIn_0.18s_ease-out]'
            >
              {currentMode.short}
            </span>
          </button>

          <div
            className={`
              absolute bottom-14 left-0 z-50 min-w-[160px] origin-bottom-left overflow-hidden
              rounded-2xl border border-primary-blue/15 bg-white shadow-lg
              transition-all duration-200 ease-out
              ${
                showModeMenu
                  ? 'pointer-events-auto translate-y-0 scale-100 opacity-100'
                  : 'pointer-events-none translate-y-2 scale-95 opacity-0'
              }
            `}
          >
            <div className='border-b border-primary-blue/10 px-3 py-2 text-xs font-medium uppercase tracking-wide text-main-text/50'>
              Mode
            </div>

            {modes.map(mode => {
              const active = selectedMode === mode.value;

              return (
                <button
                  key={mode.value}
                  type='button'
                  onClick={() => {
                    setSelectedMode(mode.value);
                    setShowModeMenu(false);
                  }}
                  className={`
                    flex w-full items-center justify-between px-3 py-2 text-sm
                    transition-all duration-150
                    ${active
                      ? 'bg-primary-blue/10 text-primary-blue'
                      : 'text-main-text hover:bg-primary-blue/5'
                    }
                  `}
                >
                  <span>{mode.label}</span>
                  <span
                    className={`
                      text-xs font-medium transition-all duration-200
                      ${active ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-1'}
                    `}
                  >
                    Active
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className='flex-1 rounded-3xl bg-primary-blue/35 p-1.5 z-50 font-mono origin-bottom animate-chat duration-400'>
          <div className='relative shrink-0 overflow-hidden rounded-3xl bg-white pr-0.5 ring-1 ring-primary-blue transition-all focus-within:ring-2'>
            <div className='absolute left-4 top-2 z-10'>
              <span
                key={currentMode.value}
                className='inline-block rounded-full bg-primary-blue/10 px-2 py-0.5 text-[11px] font-medium text-primary-blue animate-[fadeIn_0.18s_ease-out]'
              >
                {currentMode.label}
              </span>
            </div>

            <textarea
              className='block w-full max-h-[140px] resize-none rounded-3xl bg-white px-4 pt-8 pb-2 pr-11 placeholder:text-primary-blue placeholder:leading-4 focus:outline-none sm:placeholder:leading-normal'
              ref={textareaRef}
              rows='1'
              value={newMessage}
              onChange={e => setNewMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder='Ask anything'
            />

            <button
              type='button'
              className='absolute top-1/2 right-3 -translate-y-1/2 rounded-md p-1 transition-colors duration-150 hover:bg-primary-blue/20 disabled:opacity-50'
              onClick={submitNewMessage}
              disabled={isLoading}
            >
              <img src={sendIcon} alt='send' />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatInput;