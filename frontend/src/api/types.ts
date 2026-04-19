// Goal types
export type GoalStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';

export interface Goal {
  id: number;
  goal: string;
  details: string | null;
  status: GoalStatus;
  created_at: string | null;
  completed_at: string | null;
}

export interface GoalUpdate {
  status?: GoalStatus;
  details?: string;
}

// Schedule types
export type PromptType = 'check_in' | 'reminder' | 'follow_up';
export type ScheduleStatus = 'pending' | 'executed' | 'failed' | 'cancelled';

export interface ScheduledPrompt {
  id: number;
  prompt: string;
  scheduled_for: string | null;
  prompt_type: PromptType;
  goal_id: number | null;
  status: ScheduleStatus;
  executed_at: string | null;
  created_at: string | null;
}

// History types
export interface HistoryEntry {
  agent: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string | null;
}

// Chat / Orchestrator result types
export interface ConversationResult {
  type: 'conversation';
  response: string;
  signature: string;
}

export interface ClarificationResult {
  type: 'clarification';
  response: string;
  goals: string[];
  signature: string;
}

export interface ImmediateResult {
  type: 'immediate';
  goals: string[];
  clarification: string | null;
  results: Array<{ goal: string; result: unknown }>;
  message: string;
}

export interface GoalsScheduledResult {
  type: 'goals_scheduled';
  goals: string[];
  scheduled_prompts: number;
  message: string;
  immediate_results?: Array<{ goal: string; result: unknown }>;
  clarification?: string;
}

export interface ExecutedResult {
  type: 'executed';
  action: string | null;
  tool: string | null;
  parameters: Record<string, unknown>;
  result: string;
  tool_result?: unknown;
}

export interface ReminderResult {
  type: 'reminder';
  message: string;
}

export interface ErrorResponse {
  type?: undefined;
  error: string;
  raw?: string;
}

export type ChatResult =
  | ConversationResult
  | ClarificationResult
  | ImmediateResult
  | GoalsScheduledResult
  | ExecutedResult
  | ReminderResult
  | ErrorResponse;

// Chat message (local state)
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  result?: ChatResult;
  timestamp: Date;
  agent?: string;
}

// WebSocket event
export interface WsEvent {
  event: string;
  data: unknown;
}

// Agent names and colors
export const AGENT_COLORS: Record<string, string> = {
  niyati: 'var(--color-agent-niyati)',
  iccha: 'var(--color-agent-iccha)',
  karya: 'var(--color-agent-karya)',
  karma: 'var(--color-agent-karma)',
  user: 'var(--color-primary)',
};

export const STATUS_COLORS: Record<string, string> = {
  pending: 'var(--color-accent-amber)',
  in_progress: 'var(--color-accent-blue)',
  completed: 'var(--color-accent-green)',
  cancelled: 'var(--color-text-muted)',
  executed: 'var(--color-accent-green)',
  failed: 'var(--color-accent-red)',
};