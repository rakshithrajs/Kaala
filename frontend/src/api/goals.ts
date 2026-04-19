import { api } from './client';
import type { Goal, GoalUpdate } from './types';

export const goalsApi = {
  list: (status?: string) =>
    api.get('goals', { searchParams: status ? { status } : {} }).json<Goal[]>(),
  get: (id: number) =>
    api.get(`goals/${id}`).json<Goal>(),
  update: (id: number, data: GoalUpdate) =>
    api.patch(`goals/${id}`, { json: data }).json<Goal>(),
  delete: (id: number) =>
    api.delete(`goals/${id}`).json<{ ok: boolean }>(),
};