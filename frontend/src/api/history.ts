import { api } from './client';
import type { HistoryEntry } from './types';

export const historyApi = {
  list: (params?: { agent?: string; limit?: number }) =>
    api.get('history', { searchParams: params ?? {} }).json<HistoryEntry[]>(),
};