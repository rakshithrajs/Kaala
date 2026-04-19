import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { goalsApi } from '@/api/goals';
import type { GoalUpdate } from '@/api/types';

export function useGoals(status?: string) {
  return useQuery({
    queryKey: ['goals', status],
    queryFn: () => goalsApi.list(status),
  });
}

export function useGoal(id: number) {
  return useQuery({
    queryKey: ['goals', id],
    queryFn: () => goalsApi.get(id),
  });
}

export function useUpdateGoal() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: GoalUpdate }) =>
      goalsApi.update(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['goals'] });
    },
  });
}

export function useDeleteGoal() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => goalsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['goals'] });
    },
  });
}