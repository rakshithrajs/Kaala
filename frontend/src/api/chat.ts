import { api } from './client';
import type { ChatResult } from './types';

export const chatApi = {
  send: (message: string) =>
    api.post('chat', { json: { message } }).json<ChatResult>(),
};