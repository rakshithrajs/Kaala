import { useMutation } from '@tanstack/react-query';
import { chatApi } from '@/api/chat';
import { useChatStore } from '@/stores/chat-store';

export function useChat() {
  const { addUserAndResult, setSending } = useChatStore();

  const mutation = useMutation({
    mutationFn: chatApi.send,
    onMutate: (message) => {
      setSending(true);
      return message;
    },
    onSuccess: (result, message) => {
      addUserAndResult(message, result);
    },
    onSettled: () => {
      setSending(false);
    },
  });

  return {
    sendMessage: mutation.mutate,
    isSending: mutation.isPending,
    error: mutation.error,
  };
}