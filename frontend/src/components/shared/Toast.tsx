import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bell } from 'lucide-react';
import { useWsStore } from '@/stores/ws-store';

export function Toast() {
  const lastEvent = useWsStore((s) => s.lastEvent);
  const [visible, setVisible] = useState(false);
  const [event, setEvent] = useState(lastEvent);

  useEffect(() => {
    if (lastEvent && lastEvent !== event) {
      setEvent(lastEvent);
      setVisible(true);
      const timer = setTimeout(() => setVisible(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [lastEvent, event]);

  return (
    <AnimatePresence>
      {visible && event && (
        <motion.div
          initial={{ x: 400, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 400, opacity: 0 }}
          transition={{ type: 'spring', damping: 20 }}
          className="fixed top-4 right-4 z-50 max-w-sm p-4"
          style={{
            background: 'linear-gradient(135deg, var(--color-surface), var(--color-abyss))',
            border: '1px solid var(--color-primary-dim)',
            borderRadius: 'var(--radius)',
            boxShadow: '0 0 30px rgba(200,164,78,0.1)',
          }}
        >
          <div className="flex items-start gap-3">
            <div
              className="rounded-md p-2"
              style={{
                background: 'rgba(200,164,78,0.15)',
                border: '1px solid rgba(200,164,78,0.2)',
              }}
            >
              <Bell className="h-4 w-4" style={{ color: 'var(--color-primary)' }} />
            </div>
            <div>
              <p className="text-sm font-medium capitalize" style={{ color: 'var(--color-text)', fontFamily: 'var(--font-label)' }}>
                {event.event.replace(/_/g, ' ')}
              </p>
              <p className="text-xs mt-0.5 readout" style={{ color: 'var(--color-text-muted)' }}>
                {typeof event.data === 'object' ? JSON.stringify(event.data) : String(event.data)}
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}