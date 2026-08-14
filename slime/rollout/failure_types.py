"""Structured rollout failure classification shared by generators and filters."""

from __future__ import annotations

from enum import Enum


class FailureClass(str, Enum):
    RETRYABLE_INFRA = "retryable_infra"
    PERMANENT_TASK = "permanent_task_failure"
    POLICY = "policy_failure"


class MCPLeaseTimeout(TimeoutError):
    """Raised when no local MCP process shard becomes available in time."""
