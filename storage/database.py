"""Database configuration and session management using SQLAlchemy."""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from config.config import get_settings


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Database:
    """Database manager with async support."""

    def __init__(self, db_url: str | None = None):
        if db_url is None:
            settings = get_settings()
            db_url = f"sqlite+aiosqlite:///{settings.database_path}"

        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_tables(self):
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self):
        """Drop all tables (useful for testing)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_session(self) -> AsyncSession:
        """Get a new database session."""
        return self.async_session()


# Global database instance
_db: Database | None = None


async def get_db() -> Database:
    """Get the global database instance, initializing if needed."""
    global _db
    if _db is None:
        _db = Database()
        await _db.create_tables()
    return _db


async def init_db():
    """Initialize the database."""
    db = await get_db()
    return db