"""Database configuration and session management using SQLAlchemy."""

import logging

from sqlalchemy import inspect as sa_inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from kaala.config.settings import get_settings

logger = logging.getLogger(__name__)


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
        """Create all tables and migrate existing ones with missing columns."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await self._migrate_tables(conn)

    async def _migrate_tables(self, conn):
        """Add missing columns to existing tables (SQLite ALTER TABLE migration)."""

        def _inspect_and_migrate(sync_conn):
            inspector = sa_inspect(sync_conn)
            existing_tables = inspector.get_table_names()
            dialect = sync_conn.dialect

            migrations = []
            for table in Base.metadata.sorted_tables:
                if table.name not in existing_tables:
                    continue

                existing_columns = {
                    col["name"] for col in inspector.get_columns(table.name)
                }

                for column in table.columns:
                    if column.name not in existing_columns:
                        col_type = column.type.compile(dialect=dialect)
                        nullable = "" if column.nullable else "NOT NULL"
                        default = ""
                        if column.server_default is not None:
                            default = f" DEFAULT {column.server_default.arg}"
                        elif column.default is not None and column.default.is_scalar:
                            default = f" DEFAULT {column.default.arg}"

                        migrations.append(
                            f"ALTER TABLE {table.name} "
                            f"ADD COLUMN {column.name} {col_type} {nullable}{default}"
                        )
            return migrations

        migrations = await conn.run_sync(_inspect_and_migrate)

        for alter_sql in migrations:
            logger.info("Migrating: %s", alter_sql)
            await conn.execute(text(alter_sql))

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