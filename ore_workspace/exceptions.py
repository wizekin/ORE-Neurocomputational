"""Domain-specific exceptions raised by the shared workspace subsystem."""


class WorkspaceError(Exception):
    """Base exception for workspace lifecycle, access, and coordination failures."""


class UnknownWorkspaceError(WorkspaceError):
    """Raised when a publication references a workspace that does not exist."""


class PermissionDeniedError(WorkspaceError):
    """Raised when a producer attempts to modify a field it does not own."""


class InvalidFindingError(WorkspaceError):
    """Raised when a proposed finding cannot satisfy the common information protocol."""


class RevisionNotFoundError(WorkspaceError):
    """Raised when a caller requests a revision not retained by the workspace."""


class SchedulerLimitError(WorkspaceError):
    """Raised when module activation exceeds configured bounded-processing limits."""