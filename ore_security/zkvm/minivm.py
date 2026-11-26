"""Deterministic MiniVM: the committed-program substrate for execution proofs (step 33).

The MiniVM is a small stack machine with fixed-width integer arithmetic reduced
modulo a configurable modulus, a bounded register file and a hard step limit. A
program is a tuple of instructions plus a constant table; its canonical serialization
is hashed into a ``ProgramCommitment`` together with the runtime version and input
encoding. Determinism is total: identical programs and inputs yield identical
executions, which is what the execution-proof verifier relies on.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..canonical import canonical_bytes, digest_bytes
from ..exceptions import ExecutionProofError

RUNTIME_VERSION = "minivm-1.0.0"
DEFAULT_MODULUS = (1 << 61) - 1  # Mersenne prime 2^61 - 1
DEFAULT_MEMORY_SIZE = 32
DEFAULT_MAX_STEPS = 200_000

OPCODES = frozenset({
    "PUSHC", "PUSHP", "PUSHW", "POPM", "PUSHM", "ADD", "SUB", "MUL", "NEG",
    "LT", "EQ", "JMPIF", "JMP", "HALT", "NOP",
})


@dataclass(frozen=True, slots=True)
class Instruction:
    """One MiniVM instruction: opcode plus optional integer operand."""

    opcode: str
    operand: int = 0

    def __post_init__(self) -> None:
        if self.opcode not in OPCODES:
            raise ValueError(f"unknown opcode: {self.opcode}")
        if ("PUSHC" == self.opcode or "PUSHP" == self.opcode or "PUSHW" == self.opcode
                or "POPM" == self.opcode or "PUSHM" == self.opcode
                or "JMPIF" == self.opcode or "JMP" == self.opcode) and self.operand < 0:
            raise ValueError(f"{self.opcode} requires a non-negative operand")
        if self.opcode in {"ADD", "SUB", "MUL", "NEG", "LT", "EQ", "HALT", "NOP"} and self.operand != 0:
            raise ValueError(f"{self.opcode} takes no operand")


@dataclass(frozen=True, slots=True)
class Program:
    """A committed program: instruction tuple and constant table."""

    instructions: tuple[Instruction, ...]
    constants: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ValueError("a program must contain at least one instruction")
        for instruction in self.instructions:
            if not isinstance(instruction, Instruction):
                raise TypeError("program entries must be Instruction objects")

    def canonical(self) -> list[list]:
        return [
            [[instruction.opcode, int(instruction.operand)] for instruction in self.instructions],
            [int(value) for value in self.constants],
        ]


def program_commitment(program: Program, *, modulus: int = DEFAULT_MODULUS,
                       memory_size: int = DEFAULT_MEMORY_SIZE) -> dict[str, str]:
    """Freeze program, runtime and input encoding into a digest commitment."""
    payload = {
        "runtime_version": RUNTIME_VERSION,
        "program": program.canonical(),
        "modulus": str(modulus),
        "memory_size": memory_size,
        "input_encoding": "int-vec-little-endian-field-elements",
    }
    return {
        "program_digest": digest_bytes(canonical_bytes(payload)),
        "runtime_version": RUNTIME_VERSION,
        "modulus": str(modulus),
        "memory_size": memory_size,
    }


@dataclass
class ExecutionState:
    """Complete MiniVM state; hashed into the execution trace after each step."""

    pc: int
    stack: tuple[int, ...]
    memory: tuple[int, ...]
    halted: bool

    def canonical(self) -> list:
        return [self.pc, list(self.stack), list(self.memory), int(self.halted)]


class MiniVM:
    """Interpreter executing committed programs deterministically."""

    def __init__(self, *, modulus: int = DEFAULT_MODULUS,
                 memory_size: int = DEFAULT_MEMORY_SIZE,
                 max_steps: int = DEFAULT_MAX_STEPS) -> None:
        self.modulus = int(modulus)
        self.memory_size = int(memory_size)
        self.max_steps = int(max_steps)

    def execute(self, program: Program, public_inputs: tuple[int, ...] = (),
                private_inputs: tuple[int, ...] = (),
                *, output_cells: int = 4) -> dict:
        """Run the program and return outputs plus the per-step state-digest trace."""
        state = ExecutionState(0, (), (0,) * self.memory_size, False)
        trace: list[str] = []
        steps = 0
        while not state.halted:
            if steps >= self.max_steps:
                raise ExecutionProofError("program exceeded the step limit")
            state = self._step(program, state, public_inputs, private_inputs)
            steps += 1
            trace.append(digest_bytes(canonical_bytes(
                {"step": steps, "state": state.canonical()}
            )))
        outputs = state.memory[:max(0, output_cells)]
        return {
            "outputs": list(outputs),
            "final_state": state.canonical(),
            "final_state_digest": trace[-1] if trace else digest_bytes(canonical_bytes(
                {"step": 0, "state": state.canonical()})),
            "trace": trace,
            "steps": steps,
        }

    def _step(self, program: Program, state: ExecutionState,
              public_inputs: tuple[int, ...], private_inputs: tuple[int, ...]) -> ExecutionState:
        instruction = program.instructions[state.pc]
        opcode, operand = instruction.opcode, instruction.operand
        stack, memory, pc = list(state.stack), list(state.memory), state.pc
        halted = state.halted

        def pop() -> int:
            if not stack:
                raise ExecutionProofError("stack underflow")
            return stack.pop()

        def push(value: int) -> None:
            if len(stack) >= 256:
                raise ExecutionProofError("stack overflow")
            stack.append(value % self.modulus)

        if opcode == "PUSHC":
            if operand >= len(program.constants):
                raise ExecutionProofError("constant index out of range")
            push(program.constants[operand])
        elif opcode == "PUSHP":
            if operand >= len(public_inputs):
                raise ExecutionProofError("public input index out of range")
            push(public_inputs[operand])
        elif opcode == "PUSHW":
            if operand >= len(private_inputs):
                raise ExecutionProofError("private input index out of range")
            push(private_inputs[operand])
        elif opcode == "POPM":
            if operand >= self.memory_size:
                raise ExecutionProofError("memory write out of range")
            memory[operand] = pop()
        elif opcode == "PUSHM":
            if operand >= self.memory_size:
                raise ExecutionProofError("memory read out of range")
            push(memory[operand])
        elif opcode == "ADD":
            b, a = pop(), pop(); push(a + b)
        elif opcode == "SUB":
            b, a = pop(), pop(); push(a - b)
        elif opcode == "MUL":
            b, a = pop(), pop(); push(a * b)
        elif opcode == "NEG":
            push(-pop())
        elif opcode == "LT":
            b, a = pop(), pop(); push(1 if a < b else 0)
        elif opcode == "EQ":
            b, a = pop(), pop(); push(1 if a == b else 0)
        elif opcode == "JMPIF":
            if pop() != 0:
                if operand >= len(program.instructions):
                    raise ExecutionProofError("jump target out of range")
                pc = operand
                return ExecutionState(pc, tuple(stack), tuple(memory), halted)
        elif opcode == "JMP":
            if operand >= len(program.instructions):
                raise ExecutionProofError("jump target out of range")
            pc = operand
            return ExecutionState(pc, tuple(stack), tuple(memory), halted)
        elif opcode == "HALT":
            halted = True
        elif opcode == "NOP":
            pass
        else:  # pragma: no cover - opcode validated on construction
            raise ExecutionProofError(f"unreachable opcode: {opcode}")
        return ExecutionState(pc + 1, tuple(stack), tuple(memory), halted)
