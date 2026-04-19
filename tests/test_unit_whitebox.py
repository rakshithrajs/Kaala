"""White-box unit tests for internal logic and utility modules."""

import json
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy
import pytest
from pydantic import ValidationError

from kaala.agent.factory import AgentFactory
from kaala.config.settings import Settings, get_settings
from kaala.core.orchestrator import Orchestrator
from kaala.core.tools import Tool, ToolRegistry
from kaala.tools.message import MessageTool
from kaala.utils.custom_errors import AgentError, FileError
from kaala.utils.model_parser import model_select
from kaala.utils.prompt_loaders import load_prompt
from kaala.utils.response_templates import (
    IcchaResponse,
    KarmaResponse,
    KaryaResponse,
    NiyatiResponse,
    NormalResponse,
)


# ─── Orchestrator._parse_json ───────────────────────────────────────────────


class TestOrchestratorParseJson:
    """Test Orchestrator._parse_json with various LLM output formats."""

    def setup_method(self):
        self.orch = Orchestrator.__new__(Orchestrator)

    def test_valid_json_object(self):
        result = self.orch._parse_json('{"route_to": "Iccha", "user_prompt": "test"}')
        assert result == {"route_to": "Iccha", "user_prompt": "test"}

    def test_valid_json_array(self):
        result = self.orch._parse_json('[{"goal_detected": true}]')
        assert result == [{"goal_detected": True}]

    def test_none_input(self):
        assert self.orch._parse_json(None) is None

    def test_invalid_json(self):
        assert self.orch._parse_json("not json at all") is None

    def test_markdown_code_block_json(self):
        raw = '```json\n{"route_to": "Karma"}\n```'
        result = self.orch._parse_json(raw)
        assert result == {"route_to": "Karma"}

    def test_markdown_code_block_no_language(self):
        raw = '```\n{"route_to": "Iccha"}\n```'
        result = self.orch._parse_json(raw)
        assert result == {"route_to": "Iccha"}

    def test_empty_string(self):
        assert self.orch._parse_json("") is None

    def test_whitespace_json(self):
        result = self.orch._parse_json('  {"key": "value"}  ')
        assert result == {"key": "value"}

    def test_nested_json(self):
        data = {"action": "remind", "parameters": {"message": "hi", "time": "2025-01-01"}}
        result = self.orch._parse_json(json.dumps(data))
        assert result == data


# ─── Orchestrator._as_list ──────────────────────────────────────────────────


class TestOrchestratorAsList:
    def setup_method(self):
        self.orch = Orchestrator.__new__(Orchestrator)

    def test_list_passthrough(self):
        data = [{"a": 1}, {"b": 2}]
        assert self.orch._as_list(data) == data

    def test_dict_wrapped(self):
        data = {"a": 1}
        assert self.orch._as_list(data) == [{"a": 1}]

    def test_none_returns_empty(self):
        assert self.orch._as_list(None) == []

    def test_string_wrapped(self):
        assert self.orch._as_list("hello") == ["hello"]


# ─── Orchestrator._as_dict ─────────────────────────────────────────────────


class TestOrchestratorAsDict:
    def setup_method(self):
        self.orch = Orchestrator.__new__(Orchestrator)

    def test_dict_passthrough(self):
        data = {"route_to": "Iccha"}
        assert self.orch._as_dict(data) == data

    def test_single_item_list_unwrapped(self):
        data = [{"route_to": "Karma"}]
        assert self.orch._as_dict(data) == {"route_to": "Karma"}

    def test_empty_list_returns_none(self):
        assert self.orch._as_dict([]) is None

    def test_none_returns_none(self):
        assert self.orch._as_dict(None) is None

    def test_list_of_non_dicts_returns_none(self):
        assert self.orch._as_dict(["a", "b"]) is None

    def test_multi_item_list_returns_first(self):
        data = [{"a": 1}, {"b": 2}]
        assert self.orch._as_dict(data) == {"a": 1}


# ─── Response Templates ────────────────────────────────────────────────────


class TestNormalResponse:
    def test_defaults(self):
        r = NormalResponse(response="Hello")
        assert r.response == "Hello"
        assert r.signature == "Normal"

    def test_custom_signature(self):
        r = NormalResponse(response="Hi", signature="Bot")
        assert r.signature == "Bot"


class TestNiyatiResponse:
    def test_valid_routes(self):
        for route in ("Iccha", "Karya", "Karma"):
            r = NiyatiResponse(route_to=route, user_prompt="test")
            assert r.route_to == route

    def test_invalid_route(self):
        with pytest.raises(ValidationError):
            NiyatiResponse(route_to="Invalid", user_prompt="test")

    def test_defaults(self):
        r = NiyatiResponse(route_to="Iccha", user_prompt="hello")
        assert r.signature == "Niyati"


class TestIcchaResponse:
    def test_goal_detected_true(self):
        r = IcchaResponse(goal_detected=True, goal="Learn Python", response="Great idea!")
        assert r.goal_detected is True
        assert r.goal == "Learn Python"
        assert r.urgency == "later"
        assert r.needs_clarification is False

    def test_no_goal(self):
        r = IcchaResponse(goal_detected=False, response="Just chatting")
        assert r.goal is None
        assert r.details is None

    def test_urgency_immediate(self):
        r = IcchaResponse(goal_detected=True, goal="Fire!", urgency="immediate")
        assert r.urgency == "immediate"


class TestKaryaResponse:
    def test_basic_creation(self):
        r = KaryaResponse(
            goal="Learn guitar",
            prompt="Practice guitar for 30 min",
            timestamp=datetime(2025, 6, 15, 10, 0),
        )
        assert r.goal == "Learn guitar"
        assert r.prompt_type == "check_in"

    def test_timestamp_from_string(self):
        r = KaryaResponse(
            goal="Read",
            prompt="Read book",
            timestamp="2025-06-15T10:00:00",
        )
        assert isinstance(r.timestamp, datetime)

    def test_timestamp_from_numpy(self):
        ts = numpy.datetime64("2025-06-15T10:00:00")
        r = KaryaResponse(
            goal="Read",
            prompt="Read book",
            timestamp=ts.astype("datetime64[us]").astype(datetime),
        )
        assert isinstance(r.timestamp, datetime)

    def test_custom_prompt_type(self):
        r = KaryaResponse(
            goal="Exercise",
            prompt="Go for a run",
            timestamp=datetime.now(),
            prompt_type="follow_up",
        )
        assert r.prompt_type == "follow_up"


class TestKarmaResponse:
    def test_basic_creation(self):
        r = KarmaResponse(action="remind", tool="reminder", parameters={"message": "hi"})
        assert r.action == "remind"
        assert r.tool == "reminder"
        assert r.parameters == {"message": "hi"}

    def test_default_parameters(self):
        r = KarmaResponse(action="respond", tool="message")
        assert r.parameters == {}

    def test_signature_default(self):
        r = KarmaResponse(action="a", tool="t")
        assert r.signature == "Karma"


# ─── Custom Errors ──────────────────────────────────────────────────────────


class TestCustomErrors:
    def test_file_error(self):
        err = FileError("file not found")
        assert str(err) == "file not found"
        assert err.message == "file not found"

    def test_agent_error_default_message(self):
        err = AgentError()
        assert "niyati" in str(err).lower()

    def test_agent_error_custom_message(self):
        err = AgentError("custom error")
        assert str(err) == "custom error"

    def test_agent_error_is_exception(self):
        assert issubclass(AgentError, Exception)
        assert issubclass(FileError, Exception)


# ─── Model Parser ──────────────────────────────────────────────────────────


class TestModelParser:
    def test_known_model(self):
        result = model_select("GLM-5-CLOUD")
        assert result == "glm-5:cloud"

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            model_select("NONEXISTENT-MODEL")

    def test_missing_config_file(self):
        from pathlib import Path
        with patch("kaala.utils.model_parser._MODEL_FILE_PATH", Path("/nonexistent/path.json")):
            with pytest.raises(FileError):
                model_select("GLM-5-CLOUD")


# ─── Prompt Loaders ────────────────────────────────────────────────────────


class TestPromptLoaders:
    def test_known_prompts(self):
        for name in ("Iccha", "Karya", "Niyati", "Karma"):
            content = load_prompt(name)
            assert isinstance(content, str)
            assert len(content) > 0

    def test_unknown_prompt_raises(self):
        with pytest.raises(FileError):
            load_prompt("NonExistent")

    def test_prompt_content_not_empty(self):
        content = load_prompt("Niyati")
        assert len(content.strip()) > 0


# ─── Agent Factory ──────────────────────────────────────────────────────────


class TestAgentFactory:
    def test_list_agents(self):
        agents = AgentFactory.list_agents()
        assert set(agents) == {"niyati", "iccha", "karya", "karma", "normal"}

    def test_create_each_agent(self):
        for name in ("niyati", "iccha", "karya", "karma", "normal"):
            agent = AgentFactory.create(name=name, model="GLM-5-CLOUD")
            assert agent is not None

    def test_create_unknown_raises(self):
        with pytest.raises(AgentError):
            AgentFactory.create(name="unknown")

    def test_create_case_insensitive(self):
        agent = AgentFactory.create(name="NIYATI", model="GLM-5-CLOUD")
        assert "Niyati" in agent.name()

    def test_get_agent_info(self):
        agent = AgentFactory.create(name="iccha", model="GLM-5-CLOUD")
        info = AgentFactory.get_agent_info(agent)
        assert "name" in info
        assert "model" in info


# ─── Config Settings ────────────────────────────────────────────────────────


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.database_path == "kaala.db"
        assert s.default_model == "GLM-5-CLOUD"
        assert s.poll_interval == 60
        assert s.host == "127.0.0.1"
        assert s.port == 8000

    def test_from_env(self):
        env = {
            "KAALA_DB_PATH": "/tmp/test.db",
            "KAALA_DEFAULT_MODEL": "GLM-5-CLOUD",
            "KAALA_POLL_INTERVAL": "30",
            "KAALA_HOST": "0.0.0.0",
            "KAALA_PORT": "9000",
        }
        with patch.dict(os.environ, env, clear=False):
            s = Settings.from_env()
            assert s.database_path == "/tmp/test.db"
            assert s.default_model == "GLM-5-CLOUD"
            assert s.poll_interval == 30
            assert s.host == "0.0.0.0"
            assert s.port == 9000

    def test_get_settings_singleton(self):
        # Reset global
        import kaala.config.settings as cfg
        cfg._settings = None
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        # Cleanup
        cfg._settings = None


# ─── Tool Registry (additional coverage) ───────────────────────────────────


class TestToolRegistryExtended:
    def test_register_duplicate_overwrites(self):
        registry = ToolRegistry()
        tool1 = MessageTool()
        tool2 = MessageTool()
        registry.register(tool1)
        registry.register(tool2)
        assert len(registry.list_tools()) == 1

    def test_descriptions_format(self):
        registry = ToolRegistry()
        registry.register(MessageTool())
        desc = registry.tool_descriptions()
        assert "message:" in desc

    def test_abstract_tool_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Tool()


# ─── KaryaResponse timestamp edge cases ────────────────────────────────────


class TestKaryaResponseTimestampEdgeCases:
    def test_iso_format_with_timezone(self):
        r = KaryaResponse(
            goal="Test",
            prompt="Test",
            timestamp="2025-06-15T10:00:00+05:30",
        )
        assert isinstance(r.timestamp, datetime)

    def test_iso_format_without_time(self):
        r = KaryaResponse(
            goal="Test",
            prompt="Test",
            timestamp="2025-06-15",
        )
        assert isinstance(r.timestamp, datetime)

    def test_datetime_object_passthrough(self):
        dt = datetime(2025, 6, 15, 10, 0, 0)
        r = KaryaResponse(
            goal="Test",
            prompt="Test",
            timestamp=dt,
        )
        assert r.timestamp == dt