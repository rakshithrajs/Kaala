import { create } from 'zustand';
import type { ChatMessage, ChatResult } from '@/api/types';

function extractContent(result: ChatResult): string {
  if ('error' in result) return result.error;
  if (result.type === 'conversation') return result.response;
  if (result.type === 'clarification') return result.response;
  if (result.type === 'reminder') return result.message;
  if (result.type === 'executed') return result.result;
  if (result.type === 'immediate') return result.message;
  if (result.type === 'goals_scheduled') return result.message;
  return '';
}

interface ChatState {
  messages: ChatMessage[];
  isSending: boolean;
  addMessage: (msg: ChatMessage) => void;
  addUserAndResult: (content: string, result: ChatResult, agent?: string) => void;
  setSending: (v: boolean) => void;
  clearMessages: () => void;
}

let msgCounter = 0;

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isSending: false,

  addMessage: (msg) =>
    set((state) => ({ messages: [...state.messages, msg] })),

  addUserAndResult: (content, result, agent) =>
    set((state) => {
      const now = new Date();
      const userMsg: ChatMessage = {
        id: `msg-${++msgCounter}`,
        role: 'user',
        content,
        timestamp: now,
      };
      const assistantMsg: ChatMessage = {
        id: `msg-${++msgCounter}`,
        role: 'assistant',
        content: extractContent(result),
        result,
        timestamp: now,
        agent,
      };
      return { messages: [...state.messages, userMsg, assistantMsg] };
    }),

  setSending: (v) => set({ isSending: v }),
  clearMessages: () => set({ messages: [] }),
}));