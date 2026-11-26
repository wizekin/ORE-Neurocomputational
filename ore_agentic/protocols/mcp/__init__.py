"""MCP adapters and capability policy (Steps 31-32, 34)."""
from __future__ import annotations

from .client import (
    CapabilityNotAllowedError,
    MalformedResponseError,
    MCPClientAdapter,
    MCPEndpoint,
    normalize_response,
)
from .server import (
    MCPCapability,
    MCPCapabilityCatalog,
    MCPServerAdapter,
    build_readonly_mcp_catalog,
)

__all__ = (
    "CapabilityNotAllowedError",
    "MalformedResponseError",
    "MCPCapability",
    "MCPCapabilityCatalog",
    "MCPClientAdapter",
    "MCPEndpoint",
    "MCPServerAdapter",
    "build_readonly_mcp_catalog",
    "normalize_response",
)
