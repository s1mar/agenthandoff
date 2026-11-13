"""Errors that mean "the local model cannot do this job right now".

Every one of them makes the CLI exit with code 2 and the MCP tool return isError,
with a message telling the agent to do the job itself. Keeping them under one base
class means callers catch one thing, and a new failure kind cannot slip through.
"""


class Unavailable(Exception):
    """Base class. The message says what went wrong and, where possible, how to fix it."""


class NotConfigured(Unavailable):
    """No backend is set up yet (no config file and no environment variables)."""


class BadSetting(Unavailable):
    """A setting has a value we cannot use."""


class BackendError(Unavailable):
    """The model server could not be reached, failed to start, or sent a bad reply."""


class ThinkingOnly(BackendError):
    """The model spent its whole reply thinking and wrote no answer."""


class LockBusy(Unavailable):
    """Another agenthandoff job holds the GPU lock."""


class GpuBusy(Unavailable):
    """Another program (not using our lock) is using the GPU."""


class InputTooLarge(Unavailable):
    """The files asked about are bigger than one digest is allowed to read."""


class NoInput(Unavailable):
    """None of the given paths had readable text (missing, binary, unreadable, too big)."""


class ToolFailed(Unavailable):
    """A code tool or a user script could not give an answer (failed, timed out, printed nothing)."""


class RemoteRefused(Unavailable):
    """A provider would send file text off this machine, and allow_remote is off."""
