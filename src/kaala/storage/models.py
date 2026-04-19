"""SQLAlchemy database models for Kaala."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from kaala.storage.database import Base


class Goal(Base):
    """Represents a goal extracted by Iccha."""

    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    goal_text: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, in_progress, completed, cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationship to scheduled prompts
    scheduled_prompts: Mapped[list["ScheduledPrompt"]] = relationship(
        back_populates="goal", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Goal(id={self.id}, goal_text='{self.goal_text[:50]}...', status='{self.status}')>"


class ScheduledPrompt(Base):
    """Represents a prompt scheduled by Karya for future execution."""

    __tablename__ = "scheduled_prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    goal_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("goals.id"), nullable=True
    )
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_type: Mapped[str] = mapped_column(
        String(50), default="check_in"
    )  # check_in, reminder, follow_up
    scheduled_for: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, executed, failed, cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to goal
    goal: Mapped[Optional["Goal"]] = relationship(back_populates="scheduled_prompts")

    def __repr__(self) -> str:
        return f"<ScheduledPrompt(id={self.id}, scheduled_for='{self.scheduled_for}', status='{self.status}', type='{self.prompt_type}')>"


class ConversationHistory(Base):
    """Stores conversation history for context persistence."""

    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<ConversationHistory(id={self.id}, agent='{self.agent_name}', role='{self.role}')>"


class UserContext(Base):
    """Stores user preferences and context."""

    __tablename__ = "user_context"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<UserContext(key='{self.key}', value='{self.value[:50]}...')>"