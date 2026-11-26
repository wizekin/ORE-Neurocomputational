"""Distributed task scheduling with leases and fencing epochs (Step 40).

A coordinator assigns a task with task ID, owner, lease expiry, attempt
number, and fencing epoch; a worker validates current ownership before
committing outputs; timeout reassignment advances ownership only through the
authoritative state machine, so stale workers can never overwrite newer
results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ore_distributed.consensus.raft import RaftCluster
from ore_distributed.consensus.state_machine import StateMachineCommand


class StaleOwner(RuntimeError):
    """Raised when a worker writes without current fenced ownership."""


@dataclass(slots=True)
class TaskRecord:
    """The authoritative task assignment record of one task (Step 40)."""

    task_id: str
    owner: str
    lease_expiry: float
    attempt: int
    fencing_epoch: int

    def __post_init__(self) -> None:
        """Validate the task record fields."""
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.owner.strip():
            raise ValueError("owner must be non-empty")
        if self.attempt < 1:
            raise ValueError("attempt numbering starts at 1")
        if self.fencing_epoch < 1:
            raise ValueError("fencing_epoch starts at 1")


class TaskCoordinator:
    """Assign and reassign tasks strictly through the authoritative path (Step 40).

    The coordinator wraps a consensus cluster: assignments and reassignments
    are committed commands, ownership validation on commit is checked against
    the authoritative task record, and reassignment on timeout happens only
    after the state machine advances ownership.
    """

    def __init__(self, cluster: RaftCluster, clock: Callable[[], float] = lambda: 0.0) -> None:
        """Create a coordinator over one consensus cluster."""
        self.cluster = cluster
        self._clock = clock
        self._command_seq = 0
        self.stale_rejections = 0

    def _leader(self):
        """Return the current leader or raise."""
        leader = self.cluster.leader()
        if leader is None:
            raise RuntimeError("no stable leader; cannot assign tasks")
        return leader

    def _next_command(self, task_id: str, action: str) -> str:
        """Return a fresh command identifier for one task action."""
        self._command_seq += 1
        return f"task-{action}-{task_id}-{self._command_seq}"

    def assign(
        self,
        task_id: str,
        owner: str,
        lease_expiry: float,
        attempt: int = 1,
    ) -> TaskRecord:
        """Assign a task through the authoritative state machine."""
        leader = self._leader()
        command = StateMachineCommand(
            command_id=self._next_command(task_id, "assign"),
            command_type="assign_task",
            payload={
                "task_id": task_id,
                "owner": owner,
                "lease_expiry": lease_expiry,
                "attempt": attempt,
            },
        )
        leader.propose(command)
        self.cluster.run_until_committed(command.command_id, max_ticks=12)
        # Followers catch up through heartbeats before the caller reads state.
        self.cluster.run_ticks(4)
        record = self._task_from_state(task_id)
        if record is None:
            raise RuntimeError("assignment failed to commit")
        return record

    def _task_from_state(self, task_id: str) -> TaskRecord | None:
        """Read the authoritative task record from consensus state."""
        leader = self._leader()
        tasks = leader.state_machine.tasks.get(task_id)
        if tasks is None:
            return None
        return TaskRecord(
            task_id=task_id,
            owner=str(tasks.get("owner", "")),
            lease_expiry=float(tasks.get("lease_expiry", 0.0)),
            attempt=int(tasks.get("attempt", 1)),
            fencing_epoch=int(tasks.get("fencing_epoch", 1)),
        )

    def current_ownership(self, task_id: str) -> TaskRecord | None:
        """Return the authoritative ownership record of a task."""
        return self._task_from_state(task_id)

    def validate_worker_output(
        self,
        task_id: str,
        owner: str,
        fencing_epoch: int,
    ) -> bool:
        """Validate ownership before committing worker outputs (Step 40)."""
        record = self.current_ownership(task_id)
        if record is None:
            self.stale_rejections += 1
            raise StaleOwner(f"task {task_id!r} has no authoritative assignment")
        if record.owner != owner or record.fencing_epoch > fencing_epoch:
            self.stale_rejections += 1
            raise StaleOwner(
                f"worker {owner!r} holds stale fencing epoch {fencing_epoch}; "
                f"authoritative record is epoch {record.fencing_epoch} owner {record.owner!r}"
            )
        return True

    def commit_output(
        self,
        task_id: str,
        owner: str,
        fencing_epoch: int,
        output_digest: str,
    ) -> str:
        """Commit task output after ownership validation with fencing."""
        self.validate_worker_output(task_id, owner, fencing_epoch)
        leader = self._leader()
        command = StateMachineCommand(
            command_id=self._next_command(task_id, "complete"),
            command_type="complete_task",
            payload={
                "task_id": task_id,
                "owner": owner,
                "fencing_epoch": fencing_epoch,
                "output_digest": output_digest,
            },
        )
        leader.propose(command)
        self.cluster.run_until_committed(command.command_id, max_ticks=12)
        self.cluster.run_ticks(4)
        return command.command_id

    def reassign_after_timeout(
        self,
        task_id: str,
        new_owner: str,
        lease_expiry: float,
    ) -> TaskRecord:
        """Reassign a timed-out task only through the state machine (Step 40)."""
        record = self._task_from_state(task_id)
        if record is None:
            raise ValueError(f"task {task_id!r} is unknown")
        leader = self._leader()
        command = StateMachineCommand(
            command_id=self._next_command(task_id, "reassign"),
            command_type="assign_task",
            payload={
                "task_id": task_id,
                "owner": new_owner,
                "lease_expiry": lease_expiry,
                "attempt": record.attempt + 1,
            },
        )
        leader.propose(command)
        self.cluster.run_until_committed(command.command_id, max_ticks=12)
        self.cluster.run_ticks(4)
        return self._task_from_state(task_id)  # type: ignore[return-value]


def stale_worker_cannot_overwrite(
    coordinator: TaskCoordinator,
    task_id: str,
    stale_owner: str,
    stale_epoch: int,
    digest: str,
) -> tuple[bool, str]:
    """Prove a stale worker's write is rejected by fencing (Step 40 check)."""
    try:
        coordinator.commit_output(task_id, stale_owner, stale_epoch, digest)
        return False, "stale write unexpectedly accepted"
    except StaleOwner as exc:
        return True, str(exc)


def delayed_worker_write_check(
    coordinator: TaskCoordinator,
    task_id: str,
    delayed_owner: str,
    delayed_epoch: int,
) -> tuple[bool, str]:
    """Check that a delayed worker cannot overwrite a newer task result."""
    return stale_worker_cannot_overwrite(
        coordinator, task_id, delayed_owner, delayed_epoch, "delayed-output-digest"
    )


__all__ = [
    "StaleOwner",
    "TaskRecord",
    "TaskCoordinator",
    "stale_worker_cannot_overwrite",
    "delayed_worker_write_check",
]
