"""Literal executable matching for ``approvals.confirm``.

Pure matching lives separately from the approval gate so the September 2026
approval-module decomposition stays intact.
"""

import ast
from collections.abc import Mapping
import shlex

from tools import approval_context
from tools.approval_detection import (
    _deobfuscate_shell_word_for_detection,
    _iter_shell_command_word_spans,
    _normalize_command_for_detection,
    _read_shell_word,
)


def _executable_basename(value: str) -> str:
    """Return a case-folded executable basename, or ``""`` when invalid."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    # Shell command names use POSIX separators even when Hermes itself runs on
    # Windows. Accept a backslash here as well so config parsing stays benign
    # and unsurprising across profiles copied between platforms.
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _configured_confirm_executables() -> set[str]:
    """Return valid executable basenames from ``approvals.confirm``.

    The config surface is deliberately strict: only a list of mappings with a
    non-empty string ``executable`` is accepted. Malformed entries are ignored
    rather than weakening or breaking the rest of command approval.
    """
    try:
        rules = approval_context._get_approval_config().get("confirm") or []
    except Exception:
        return set()
    if not isinstance(rules, list):
        return set()

    executables: set[str] = set()
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        executable = rule.get("executable")
        if not isinstance(executable, str) or not executable.strip():
            continue
        basename = _executable_basename(executable)
        if basename:
            executables.add(basename)
    return executables


_LITERAL_SHELL_INTERPRETERS = {"bash", "dash", "ksh", "sh", "zsh"}
_MAX_LITERAL_SHELL_RECURSION = 8
_AMBIGUOUS_LITERAL_SHELL_COMMAND = object()


class UnresolvedExecutable(str):
    """Fail-closed match: the real executable is not literal (``$X``, backticks, deep nesting)."""


def _unresolved(executables: set[str]) -> str:
    return UnresolvedExecutable(min(executables))


def _literal_shell_word_value(word: str) -> str | None:
    """Decode one literal shell word without performing expansions."""
    try:
        values = shlex.split(word, posix=True)
    except ValueError:
        return None
    if len(values) != 1:
        return None
    return values[0]


def _literal_shell_command_after_c(
    command: str,
    executable_end: int,
) -> str | object | None:
    """Return a shell interpreter's literal ``-c`` command string, if any."""
    pos = executable_end
    while pos < len(command):
        word_start, word_end, word = _read_shell_word(command, pos)
        if word_start == word_end:
            return None
        option = _literal_shell_word_value(word)
        if option is None or not option.startswith("-") or option == "-":
            return None
        pos = word_end
        if "c" not in option[1:]:
            continue

        script_start, script_end, script = _read_shell_word(command, pos)
        if script_start == script_end:
            return _AMBIGUOUS_LITERAL_SHELL_COMMAND
        value = _literal_shell_word_value(script)
        return value if value is not None else _AMBIGUOUS_LITERAL_SHELL_COMMAND
    return None


def _literal_env_split_command(
    command: str,
    executable_end: int,
) -> str | None:
    """Return a literal GNU ``env -S`` split string, if present."""
    pos = executable_end
    while pos < len(command):
        word_start, word_end, word = _read_shell_word(command, pos)
        if word_start == word_end:
            return None
        option = _literal_shell_word_value(word)
        if option is None or not option.startswith("-") or option == "-":
            return None
        pos = word_end

        if option in {"-S", "--split-string"}:
            value_start, value_end, value_word = _read_shell_word(command, pos)
            if value_start == value_end:
                return None
            return _literal_shell_word_value(value_word)
        if option.startswith("--split-string="):
            return option.split("=", 1)[1]
        if option.startswith("-S") and option != "-S":
            return option[2:]
        if option == "--":
            return None
        if option in {"-u", "--unset", "-C", "--chdir"}:
            value_start, value_end, _value_word = _read_shell_word(command, pos)
            if value_start == value_end:
                return None
            pos = value_end
        elif option.startswith(("-u", "-C", "--unset=", "--chdir=")):
            continue
    return None


def _match_literal_shell_executable(
    command: str,
    executables: set[str],
    *,
    depth: int = 0,
) -> str | None:
    """Match configured executables, recursively inspecting literal ``sh -c``."""
    if depth >= _MAX_LITERAL_SHELL_RECURSION:
        # We reached this point only through a literal shell ``-c`` chain.
        # Refuse to let excessive nesting turn a configured confirmation rule
        # into a bypass: ambiguity beyond the parser bound fails closed.
        return _unresolved(executables)

    for _start, executable_end, word in _iter_shell_command_word_spans(command):
        candidate = _executable_basename(
            _deobfuscate_shell_word_for_detection(word)
        )
        if candidate.startswith(("$", "`")):
            return _unresolved(executables)
        if candidate in executables:
            return candidate
        if candidate == "env":
            split_command = _literal_env_split_command(command, executable_end)
            if split_command:
                nested_match = _match_literal_shell_executable(
                    split_command,
                    executables,
                    depth=depth + 1,
                )
                if nested_match:
                    return nested_match
        if candidate not in _LITERAL_SHELL_INTERPRETERS:
            continue

        script = _literal_shell_command_after_c(command, executable_end)
        if script is _AMBIGUOUS_LITERAL_SHELL_COMMAND:
            return _unresolved(executables)
        if not script:
            continue
        assert isinstance(script, str)
        if script.lstrip().startswith(("$", "`")):
            return _unresolved(executables)
        nested_match = _match_literal_shell_executable(
            script,
            executables,
            depth=depth + 1,
        )
        if nested_match:
            return nested_match
    return None


def _match_executable_confirm_rule(command: str) -> str | None:
    """Return the configured executable invoked by a literal shell command.

    Reuse the quote-aware command-position iterator used by dangerous-command
    detection. It understands wrapper commands, environment assignments,
    pipelines, chains, subshells, and bounded literal shell ``-c`` recursion,
    while ignoring ordinary arguments and quoted prose.
    """
    executables = _configured_confirm_executables()
    if not executables or not isinstance(command, str):
        return None

    normalized = _normalize_command_for_detection(command)
    return _match_literal_shell_executable(normalized, executables)


_CONFIRM_SUBPROCESS_APIS = {
    "run",
    "call",
    "check_call",
    "check_output",
    "Popen",
}


def _literal_ast_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_argument(call: ast.Call, name: str, position: int = 0) -> ast.AST | None:
    if len(call.args) > position:
        return call.args[position]
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _call_keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _literal_ast_argv(node: ast.AST | None) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        return None
    values = [_literal_ast_string(item) for item in node.elts]
    if any(value is None for value in values):
        return None
    return [value for value in values if value is not None]


def _literal_shell_true(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "shell":
            return (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            )
    return False


class _ExecuteCodeConfirmVisitor(ast.NodeVisitor):
    """Track trusted imports and simple literal bindings in execution order."""

    def __init__(self, executables: set[str]):
        self.executables = executables
        self.module_aliases: dict[str, str] = {}
        self.function_aliases: dict[str, tuple[str, str]] = {}
        self.literal_bindings: dict[str, ast.AST] = {}
        self.matched: str | None = None

    def _clear_binding(self, name: str) -> None:
        self.module_aliases.pop(name, None)
        self.function_aliases.pop(name, None)
        self.literal_bindings.pop(name, None)

    def _bind_literal(self, name: str, value: ast.AST) -> None:
        self._clear_binding(name)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            self.literal_bindings[name] = value
        elif isinstance(value, (ast.List, ast.Tuple)) and value.elts and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str)
            for item in value.elts
        ):
            self.literal_bindings[name] = value

    def _resolved_argument(self, node: ast.AST | None) -> ast.AST | None:
        if isinstance(node, ast.Name):
            return self.literal_bindings.get(node.id)
        return node

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".", 1)[0]
            self._clear_binding(bound_name)
            if alias.name in {"os", "subprocess"}:
                self.module_aliases[bound_name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            bound_name = alias.asname or alias.name
            self._clear_binding(bound_name)
            if module == "os" and alias.name == "system":
                self.function_aliases[bound_name] = ("os", "system")
            elif module == "subprocess" and alias.name in _CONFIRM_SUBPROCESS_APIS:
                self.function_aliases[bound_name] = ("subprocess", alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind_literal(target.id, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        if isinstance(node.target, ast.Name):
            if node.value is None:
                self._clear_binding(node.target.id)
            else:
                self._bind_literal(node.target.id, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._clear_binding(node.target.id)

    def _visit_scoped_body(
        self,
        body: list[ast.stmt],
        *,
        local_names: tuple[str, ...] = (),
    ) -> None:
        saved_modules = self.module_aliases.copy()
        saved_functions = self.function_aliases.copy()
        saved_literals = self.literal_bindings.copy()
        try:
            for name in local_names:
                self._clear_binding(name)
            for statement in body:
                if self.matched is not None:
                    break
                self.visit(statement)
        finally:
            self.module_aliases = saved_modules
            self.function_aliases = saved_functions
            self.literal_bindings = saved_literals

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_scoped_body(node.body)
        self._visit_scoped_body(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._visit_scoped_body(node.body)
        self._visit_scoped_body(node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._visit_scoped_body(node.body)
        self._visit_scoped_body(node.orelse)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._visit_scoped_body(node.body)
        self._visit_scoped_body(node.orelse)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            if case.guard is not None:
                self.visit(case.guard)
            self._visit_scoped_body(case.body)

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        self._visit_scoped_body(node.body)
        for handler in node.handlers:
            if handler.type is not None:
                self.visit(handler.type)
            self._visit_scoped_body(handler.body)
        self._visit_scoped_body(node.orelse)
        self._visit_scoped_body(node.finalbody)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_try(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_try(node)

    @staticmethod
    def _argument_names(arguments: ast.arguments) -> tuple[str, ...]:
        names = [
            item.arg
            for item in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        ]
        if arguments.vararg is not None:
            names.append(arguments.vararg.arg)
        if arguments.kwarg is not None:
            names.append(arguments.kwarg.arg)
        return tuple(names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._clear_binding(node.name)
        self._visit_scoped_body(
            node.body,
            local_names=self._argument_names(node.args),
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._clear_binding(node.name)
        self._visit_scoped_body(
            node.body,
            local_names=self._argument_names(node.args),
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._clear_binding(node.name)
        self._visit_scoped_body(node.body)

    def visit_Call(self, node: ast.Call) -> None:
        if self.matched is not None:
            return

        module_name: str | None = None
        function_name: str | None = None
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module_name = self.module_aliases.get(node.func.value.id)
            function_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            target = self.function_aliases.get(node.func.id)
            if target is not None:
                module_name, function_name = target

        if module_name == "os" and function_name == "system":
            argument = self._resolved_argument(_call_argument(node, "command"))
            shell_command = _literal_ast_string(argument)
            if shell_command is not None:
                self.matched = _match_executable_confirm_rule(shell_command)
        elif module_name == "subprocess" and function_name in _CONFIRM_SUBPROCESS_APIS:
            executable_node = self._resolved_argument(
                _call_keyword(node, "executable")
            )
            explicit_executable = _literal_ast_string(executable_node)
            candidate = _executable_basename(explicit_executable or "")
            if candidate in self.executables:
                self.matched = candidate

            argument = self._resolved_argument(_call_argument(node, "args"))
            argv = _literal_ast_argv(argument)
            if self.matched is None and argv is not None:
                shell_command = " ".join(shlex.quote(item) for item in argv)
                self.matched = _match_literal_shell_executable(
                    shell_command,
                    self.executables,
                )
            elif self.matched is None:
                literal_command = _literal_ast_string(argument)
                if literal_command is not None:
                    if _literal_shell_true(node):
                        self.matched = _match_executable_confirm_rule(literal_command)
                    else:
                        candidate = _executable_basename(literal_command)
                        if (
                            candidate in self.executables
                            and not any(ch.isspace() for ch in literal_command)
                        ):
                            self.matched = candidate

        if self.matched is None:
            self.generic_visit(node)


def _match_execute_code_confirm_rule(code: str) -> str | None:
    """Find direct literal subprocess/os.system executable invocations.

    This is intentionally a narrow, non-executing AST inspection. Dynamic
    expressions and syntactically invalid scripts are left alone; terminal()
    calls are independently guarded by the terminal tool at execution time.
    """
    executables = _configured_confirm_executables()
    if not executables or not isinstance(code, str):
        return None
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError, ValueError):
        return None

    visitor = _ExecuteCodeConfirmVisitor(executables)
    visitor.visit(tree)
    return visitor.matched
