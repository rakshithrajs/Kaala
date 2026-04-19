import { api } from './client';
import type { ScheduledPrompt } from './types';

export const schedulesApi = {
  list: (status?: string) =>
    api.get('schedules', { searchParams: status ? { status } : {} }).json<ScheduledPrompt[]>(),
  cancel: (id: number) =>
    api.post(`schedules/${id}/cancel`).json<{ ok: boolean }>(),
};