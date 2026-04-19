import { useEffect, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useWsStore } from '@/stores/ws-store';

export function useWebSocket() {
  const queryClient = useQueryClient();
  const { setConnected, pushEvent } = useWsStore();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef(0);

  useEffect(() => {
    function connect() {
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${protocol}//${location.host}/ws`);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          pushEvent(msg.event, msg.data);

          if (msg.event === 'scheduled_prompt_fired') {
            queryClient.invalidateQueries({ queryKey: ['schedules'] });
            queryClient.invalidateQueries({ queryKey: ['goals'] });
          }
          if (msg.event === 'reminder') {
            queryClient.invalidateQueries({ queryKey: ['history'] });
          }
        } catch {
          // ignore non-JSON messages
        }
      };

      ws.onclose = () => {
        setConnected(false);
        const delay = Math.min(1000 * Math.pow(2, reconnectRef.current), 30_000);
        reconnectRef.current += 1;
        setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [queryClient, setConnected, pushEvent]);
}