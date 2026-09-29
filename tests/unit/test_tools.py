from apps.api.state import AppState
from packages.agents.tools import ProtectedAction, ToolContext


def test_protected_tool_only_creates_approval():
    state = AppState()
    context = ToolContext(state)
    result = context.pause_mission(ProtectedAction(mission_id="M1",reason="Weather worsened"))
    assert result["approval_required"]
    assert result["approval"]["status"] == "PENDING"
    assert len(state.approvals) == 1


def test_tool_categories_prevent_implicit_authority():
    definitions = ToolContext(AppState()).definitions()
    assert definitions["get_fleet_state"].category == "read"
    assert definitions["simulate_mission"].category == "simulation"
    assert definitions["request_return_to_base"].category == "protected"

