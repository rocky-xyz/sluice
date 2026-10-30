"""What a run reports while it goes, for the terminal session's live panel (docs/specs/12).

Nodes report themselves: a node wrapped by `reporting` says when it starts and
when it finishes, from the thread it runs on, so a parallel analyst that is
done shows as done at once. (LangGraph's own streaming holds back a finished
parallel node until the one on the main thread is done too; measured before
task 13.2.) The LLM and Jev caches report every answer they hand out.

Events are for display only: they are not part of the state and change no
cache key. A callback that raises is logged and ignored, so a display fault
never stops a run that is being paid for.
"""

import logging
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeStarted:
    name: str


@dataclass(frozen=True)
class NodeFinished:
    name: str
    update: dict[str, Any] = field(default_factory=dict)  # what the node wrote to the state
    failed: bool = False  # it raised; the error goes on to LangGraph unchanged


@dataclass(frozen=True)
class LLMCalled:
    node: str | None  # the node that asked; None outside a reporting node
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool
    tools: tuple[str, ...] = ()  # the tools the answer asks for (tool mode)


@dataclass(frozen=True)
class JevCalled:
    input_tokens: int
    output_tokens: int
    cached: bool


Event = NodeStarted | NodeFinished | LLMCalled | JevCalled
OnEvent = Callable[[Event], None]

# The node running on this thread, so that an LLM call can say who made it.
_node: ContextVar[str | None] = ContextVar("sluice_node", default=None)


def current_node() -> str | None:
    return _node.get()


def emit(on_event: OnEvent | None, event: Event) -> None:
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:  # showing progress must not break the run
        log.warning("event callback failed on %s", type(event).__name__, exc_info=True)


def reporting(name: str, node: Callable[[Any], dict], on_event: OnEvent) -> Callable[[Any], dict]:
    """`node` unchanged, but saying when it starts and finishes."""

    def wrapped(state: Any) -> dict:
        token = _node.set(name)
        try:
            emit(on_event, NodeStarted(name))
            try:
                update = node(state)
            except BaseException:
                emit(on_event, NodeFinished(name, failed=True))
                raise
            emit(on_event, NodeFinished(name, update))
            return update
        finally:
            _node.reset(token)

    return wrapped
