import { useQuery } from '@tanstack/react-query';
import { historyApi } from '@/api/history';

export function useHistory(params?: { agent?: string; limit?: number }) {
  return useQuery({
    queryKey: ['history', params],
    queryFn: () => historyApi.list(params),
  });
}