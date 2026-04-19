import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { schedulesApi } from '@/api/schedules';

export function useSchedules(status?: string) {
  return useQuery({
    queryKey: ['schedules', status],
    queryFn: () => schedulesApi.list(status),
  });
}

export function useCancelSchedule() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => schedulesApi.cancel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['schedules'] });
    },
  });
}