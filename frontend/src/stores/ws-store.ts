import { create } from 'zustand';
import type { WsEvent } from '@/api/types';

interface WsState {
  connected: boolean;
  events: WsEvent[];
  lastEvent: WsEvent | null;
  setConnected: (v: boolean) => void;
  pushEvent: (event: string, data: unknown) => void;
}

export const useWsStore = create<WsState>((set) => ({
  connected: false,
  events: [],
  lastEvent: null,

  setConnected: (v) => set({ connected: v }),

  pushEvent: (event, data) =>
    set((state) => {
      const wsEvent: WsEvent = { event, data };
      return {
        events: [...state.events.slice(-49), wsEvent],
        lastEvent: wsEvent,
      };
    }),
}));