"""Real assertions for `mcp_tools_to_openai_tools` against a real MCP `Tool`.

CR-01 (phase 01 code review): the function once read `tool.inputSchema`
(camelCase) off each MCP `Tool`. The installed `mcp>=2.2,<3` SDK exposes
that field as the Python attribute `input_schema` -- `inputSchema` is only
a wire-serialization alias, not an attribute name -- so a bare
`tool.inputSchema` raised `AttributeError` on every real `Tool` and crashed
app startup outright. No test caught this because every other test in this
suite builds tools with hand-written doubles that already use whichever
shape the test author typed, never a real `mcp.types.Tool` instance. This
test builds one for real, the way the installed SDK does, so this class of
attribute-name drift fails loudly again if it ever recurs.
"""

from mcp.types import Tool

from spire_voice.mcp_client import mcp_tools_to_openai_tools


def test_mcp_tools_to_openai_tools_reads_real_tool_input_schema():
    tool = Tool(
        name="ha_call_service",
        description="Call a Home Assistant service.",
        inputSchema={
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
        },
    )

    openai_tools = mcp_tools_to_openai_tools([tool])

    assert openai_tools == [
        {
            "type": "function",
            "function": {
                "name": "ha_call_service",
                "description": "Call a Home Assistant service.",
                "parameters": {
                    "type": "object",
                    "properties": {"entity_id": {"type": "string"}},
                },
            },
        }
    ]


def test_mcp_tools_to_openai_tools_defaults_missing_description_to_empty_string():
    tool = Tool(name="ha_list_entities", description=None, inputSchema={"type": "object", "properties": {}})

    (openai_tool,) = mcp_tools_to_openai_tools([tool])

    assert openai_tool["function"]["description"] == ""


def test_mcp_tools_to_openai_tools_keeps_a_falsy_but_present_schema():
    """WR-01 (phase 01 code review, iteration 2): CR-01's own fallback read
    `getattr(tool, "input_schema", None) or getattr(tool, "inputSchema",
    None)`. `or` is a truthiness test, not a presence test, so a schema
    that is genuinely present but falsy -- an empty dict -- was discarded
    and replaced with `None` from the second `getattr`, which resolves to
    `None` on the installed SDK because `inputSchema` is not a real
    attribute on `Tool`. That would send `{"parameters": None}` into a live
    `tools=[...]` array instead of the empty-but-valid schema the tool
    actually declared. A sentinel-based presence check keeps `{}` intact.
    """
    tool = Tool(name="ha_list_entities", description="List entities.", inputSchema={})

    (openai_tool,) = mcp_tools_to_openai_tools([tool])

    assert openai_tool["function"]["parameters"] == {}
