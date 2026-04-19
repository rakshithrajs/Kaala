import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle } from 'lucide-react';

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = 'Confirm',
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ background: 'rgba(6,7,11,0.7)', backdropFilter: 'blur(8px)' }}
          onClick={onCancel}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="p-6 max-w-md w-full mx-4"
            style={{
              background: 'linear-gradient(145deg, rgba(17,20,30,0.98), rgba(11,13,20,0.99))',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3 mb-3">
              <div
                className="rounded-md p-2"
                style={{ background: 'rgba(248,113,113,0.15)', border: '1px solid rgba(248,113,113,0.2)' }}
              >
                <AlertTriangle className="h-5 w-5" style={{ color: 'var(--color-accent-red)' }} />
              </div>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.125rem', color: 'var(--color-text)' }}>{title}</h3>
            </div>
            <p className="text-sm readout mb-6" style={{ color: 'var(--color-text-muted)' }}>{message}</p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={onCancel}
                className="px-4 py-2 rounded-md text-sm transition-colors"
                style={{ color: 'var(--color-text-muted)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-text)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-muted)'}
              >
                Cancel
              </button>
              <button
                onClick={onConfirm}
                className="px-4 py-2 rounded-md text-sm transition-colors"
                style={{ background: 'rgba(248,113,113,0.15)', color: 'var(--color-accent-red)', border: '1px solid rgba(248,113,113,0.25)' }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(248,113,113,0.25)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(248,113,113,0.15)'}
              >
                {confirmLabel}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function useConfirmDialog() {
  const [state, setState] = useState<{
    open: boolean;
    title: string;
    message: string;
    confirmLabel: string;
    onConfirm: () => void;
  }>({ open: false, title: '', message: '', confirmLabel: 'Confirm', onConfirm: () => {} });

  const confirm = (opts: { title: string; message: string; confirmLabel?: string }) =>
    new Promise<void>((resolve) => {
      setState({
        open: true,
        title: opts.title,
        message: opts.message,
        confirmLabel: opts.confirmLabel ?? 'Confirm',
        onConfirm: () => {
          setState((s) => ({ ...s, open: false }));
          resolve();
        },
      });
    });

  const cancel = () => setState((s) => ({ ...s, open: false }));

  return { dialogState: state, confirm, cancel };
}