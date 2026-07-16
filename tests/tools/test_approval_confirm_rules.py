"""Behavioral tests for one-shot ``approvals.confirm`` executable rules."""

from __future__ import annotations

import shlex
from unittest.mock import patch

import pytest

from hermes_cli.config import DEFAULT_CONFIG
from tools import approval as approval


SESSION_KEY = "confirm-rule-test-session"
CONFIRM_KEY = "executable-confirm:aws"


@pytest.fixture(autouse=True)
def clean_approval_state(monkeypatch):
    """Start every test outside interactive/gateway/cron contexts."""
    for name in (
        "HERMES_CRON_SESSION",
        "HERMES_EXEC_ASK",
        "HERMES_GATEWAY_SESSION",
        "HERMES_INTERACTIVE",
        "HERMES_SESSION_PLATFORM",
        "HERMES_YOLO_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    approval.clear_session(SESSION_KEY)
    with approval._lock:
        original_permanent = set(approval._permanent_approved)
        approval._permanent_approved.clear()
        approval._gateway_notify_cbs.pop(SESSION_KEY, None)
        approval._gateway_queues.pop(SESSION_KEY, None)
    yield
    approval.clear_session(SESSION_KEY)
    with approval._lock:
        approval._permanent_approved.clear()
        approval._permanent_approved.update(original_permanent)
        approval._gateway_notify_cbs.pop(SESSION_KEY, None)
        approval._gateway_queues.pop(SESSION_KEY, None)


@pytest.fixture
def confirm_config(monkeypatch):
    state = {
        "mode": "off",
        "cron_mode": "approve",
        "confirm": [{"executable": "aws"}],
        "deny": [],
    }
    monkeypatch.setattr(approval, "_get_approval_config", lambda: state)
    monkeypatch.setattr(
        approval,
        "_get_cron_approval_mode",
        lambda: str(state.get("cron_mode", "deny")),
    )
    return state


def _interactive(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")


def _gateway_resolver(choice: str, seen: list[dict]):
    def notify(data: dict) -> None:
        seen.append(data)
        assert approval.resolve_gateway_approval(SESSION_KEY, choice) == 1

    return notify


def _run_in_gateway(command: str, choice: str) -> tuple[dict, list[dict]]:
    seen: list[dict] = []
    token = approval.set_current_session_key(SESSION_KEY)
    approval.register_gateway_notify(
        SESSION_KEY, _gateway_resolver(choice, seen)
    )
    try:
        result = approval.check_all_command_guards(command, "local")
    finally:
        approval.unregister_gateway_notify(SESSION_KEY)
        approval.reset_current_session_key(token)
    return result, seen


def _run_code_in_gateway(code: str, choice: str) -> tuple[dict, list[dict]]:
    seen: list[dict] = []
    token = approval.set_current_session_key(SESSION_KEY)
    approval.register_gateway_notify(
        SESSION_KEY, _gateway_resolver(choice, seen)
    )
    try:
        result = approval.check_execute_code_guard(code, "local")
    finally:
        approval.unregister_gateway_notify(SESSION_KEY)
        approval.reset_current_session_key(token)
    return result, seen


def test_default_confirm_rules_are_empty():
    assert DEFAULT_CONFIG["approvals"]["confirm"] == []


def test_executable_confirm_guard_returns_none_for_nonmatch(confirm_config):
    result = approval.check_executable_confirm_guard("echo aws")

    assert result is None


def test_executable_confirm_guard_runs_one_shot_gate(
    confirm_config, monkeypatch
):
    _interactive(monkeypatch)
    calls = []

    def approve_once(command, description, **kwargs):
        calls.append((command, description, kwargs))
        return "once"

    result = approval.check_executable_confirm_guard(
        "aws sts get-caller-identity",
        approval_callback=approve_once,
    )

    assert result["approved"] is True
    assert result["executable_confirm"] is True
    assert result["user_approved"] is True
    assert calls[0][2]["allow_permanent"] is False


@pytest.mark.parametrize(
    "command",
    [
        "aws s3 ls",
        "AWS S3 LS",
        "/usr/local/bin/aws s3 ls",
        "./tools/aws s3 ls",
        "../bin/AwS s3 ls",
        "sudo -u root aws s3 ls",
        "env AWS_PROFILE=prod aws s3 ls",
        "exec aws s3 ls",
        "nohup aws s3 ls",
        "nohup -- aws s3 ls",
        "time aws s3 ls",
        "command aws s3 ls",
        "command -p aws s3 ls",
        "env -S 'AWS_PROFILE=prod aws s3 ls'",
        "env -u AWS_DEFAULT_REGION -S 'AWS_PROFILE=prod aws s3 ls'",
        "env -C /tmp aws s3 ls",
        "env --split-string='AWS_PROFILE=prod aws s3 ls'",
        "AWS_PROFILE=prod aws s3 ls",
        "echo ready && aws s3 ls",
        "printf data | aws s3 cp - s3://bucket/key",
        "(aws sts get-caller-identity)",
        "echo $(aws sts get-caller-identity)",
        r"a\ws s3 ls",
        "a''ws s3 ls",
    ],
)
def test_terminal_matches_only_executables_at_shell_command_positions(
    confirm_config, command
):
    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert "BLOCKED" in result["message"]
    assert "aws" in result["message"].lower()


@pytest.mark.parametrize(
    "command",
    [
        "echo aws",
        "grep aws file.txt",
        "printf '%s' 'aws s3 ls'",
        "echo 'run (aws s3 ls) later'",
        "aws-vault exec prod",
        "myaws s3 ls",
        "/tmp/aws-helper s3 ls",
        "echo command -p aws s3 ls",
        "printf '%s' 'nohup -- aws s3 ls'",
        "command -p printf '%s' aws",
        "env -S 'printf %s aws'",
    ],
)
def test_terminal_does_not_match_arguments_prose_or_prefixed_names(
    confirm_config, command
):
    result = approval.check_all_command_guards(command, "local")

    assert result == {"approved": True, "message": None}


def test_non_aws_wrapper_commands_still_honor_yolo(confirm_config, monkeypatch):
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)

    result = approval.check_all_command_guards(
        "nohup -- env -S 'printf %s aws'", "local"
    )

    assert result == {"approved": True, "message": None}


@pytest.mark.parametrize(
    "command",
    [
        'bash -c "aws s3 ls"',
        "sh -lc 'aws sts get-caller-identity'",
        "dash -ec 'aws s3 ls'",
        "zsh -l -c 'aws s3 ls'",
        "/bin/ksh -c 'aws sts get-caller-identity'",
        "env AWS_PROFILE=prod bash -lc 'aws s3 ls'",
        "sudo -u root sh -c 'env AWS_PROFILE=prod aws s3 ls'",
        "bash -c 'sh -lc \"aws sts get-caller-identity\"'",
    ],
)
def test_terminal_inspects_literal_shell_interpreter_commands(
    confirm_config, command
):
    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert "BLOCKED" in result["message"]
    assert "aws" in result["message"].lower()


@pytest.mark.parametrize(
    "command",
    [
        'echo "bash -c aws"',
        "printf '%s' 'sh -lc aws'",
        "bash 'aws s3 ls'",
        "sh -l 'aws s3 ls'",
        "bash -c 'echo aws'",
    ],
)
def test_terminal_does_not_scan_shell_prose_scripts_or_dynamic_commands(
    confirm_config, command
):
    result = approval.check_all_command_guards(command, "local")

    assert result == {"approved": True, "message": None}


def test_terminal_shell_literal_recursion_is_bounded_and_fails_closed(confirm_config):
    command = "aws s3 ls"
    for index in range(approval._MAX_LITERAL_SHELL_RECURSION + 1):
        shell = "bash" if index % 2 else "sh"
        command = f"{shell} -c {shlex.quote(command)}"

    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert result.get("executable_confirm") is True


@pytest.mark.parametrize(
    "command",
    [
        "$AWS_COMMAND",
        "bash -c '$AWS_COMMAND'",
        "bash -c '$(cat /tmp/command)'",
        "bash -c 'aws s3 ls",
    ],
)
def test_ambiguous_shell_command_positions_fail_closed(confirm_config, command):
    result = approval.check_all_command_guards(command, "local")

    assert result["approved"] is False
    assert result.get("executable_confirm") is True


def test_malformed_confirm_entries_are_ignored_safely(
    confirm_config, monkeypatch
):
    confirm_config["confirm"] = [
        None,
        "aws",
        17,
        {},
        {"executable": None},
        {"executable": ""},
        {"executable": "   "},
        {"other": "aws"},
    ]

    assert approval.check_all_command_guards("aws s3 ls", "local") == {
        "approved": True,
        "message": None,
    }

    confirm_config["confirm"].append({"executable": "aws"})
    blocked = approval.check_all_command_guards("aws s3 ls", "local")
    assert blocked["approved"] is False


def test_missing_confirm_is_a_noop(confirm_config):
    confirm_config.pop("confirm")
    assert approval.check_all_command_guards("aws s3 ls", "local") == {
        "approved": True,
        "message": None,
    }


def test_hardline_remains_first(confirm_config):
    result = approval.check_all_command_guards("aws s3 ls; rm -rf /", "local")

    assert result["approved"] is False
    assert result.get("hardline") is True
    assert result.get("executable_confirm") is None


def test_user_deny_remains_before_confirm(confirm_config):
    confirm_config["deny"] = ["aws*"]
    result = approval.check_all_command_guards("aws s3 ls", "local")

    assert result["approved"] is False
    assert result.get("user_deny") is True
    assert result.get("executable_confirm") is None


@pytest.mark.parametrize("entrypoint", ["combined", "dangerous"])
def test_mode_off_still_requires_interactive_approval(
    confirm_config, monkeypatch, entrypoint
):
    _interactive(monkeypatch)
    calls: list[tuple[str, str, dict]] = []

    def approve_once(command, description, **kwargs):
        calls.append((command, description, kwargs))
        return "once"

    if entrypoint == "combined":
        result = approval.check_all_command_guards(
            "aws s3 ls", "local", approval_callback=approve_once
        )
    else:
        result = approval.check_dangerous_command(
            "aws s3 ls", "local", approval_callback=approve_once
        )

    assert result["approved"] is True
    assert result.get("user_approved") is True
    assert len(calls) == 1
    assert calls[0][2]["allow_permanent"] is False
    assert calls[0][2]["one_shot_only"] is True


def test_fallback_interactive_prompt_only_offers_once_and_deny(capsys):
    with patch("builtins.input", return_value="session"):
        choice = approval.prompt_dangerous_approval(
            "aws s3 ls",
            "executable confirmation",
            allow_permanent=False,
            one_shot_only=True,
        )

    rendered = capsys.readouterr().out
    assert choice == "deny"
    assert "[o]nce" in rendered and "[d]eny" in rendered
    assert "[s]ession" not in rendered and "[a]lways" not in rendered


@pytest.mark.parametrize("bypass", ["process", "session"])
def test_process_and_session_yolo_do_not_bypass_confirm(
    confirm_config, monkeypatch, bypass
):
    _interactive(monkeypatch)
    token = approval.set_current_session_key(SESSION_KEY)
    try:
        if bypass == "process":
            monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
        else:
            approval.enable_session_yolo(SESSION_KEY)
        result = approval.check_all_command_guards(
            "aws s3 ls",
            "local",
            approval_callback=lambda *_args, **_kwargs: "deny",
        )
    finally:
        approval.reset_current_session_key(token)

    assert result["approved"] is False
    assert "denied" in result["message"].lower()


def test_session_and_permanent_allowlists_do_not_bypass_confirm(
    confirm_config, monkeypatch
):
    _interactive(monkeypatch)
    token = approval.set_current_session_key(SESSION_KEY)
    approval.approve_session(SESSION_KEY, CONFIRM_KEY)
    approval.approve_permanent("aws s3 ls")
    try:
        result = approval.check_all_command_guards(
            "aws s3 ls",
            "local",
            approval_callback=lambda *_args, **_kwargs: "deny",
        )
    finally:
        approval.reset_current_session_key(token)

    assert result["approved"] is False


def test_confirm_runs_before_smart_approval(confirm_config, monkeypatch):
    confirm_config["mode"] = "smart"
    _interactive(monkeypatch)
    smart_calls: list[str] = []
    monkeypatch.setattr(
        approval,
        "_smart_approve",
        lambda command, _description: smart_calls.append(command) or "approve",
    )

    result = approval.check_all_command_guards(
        "aws s3 ls",
        "local",
        approval_callback=lambda *_args, **_kwargs: "deny",
    )

    assert result["approved"] is False
    assert smart_calls == []


@pytest.mark.parametrize("choice, expected", [("once", True), ("deny", False)])
def test_gateway_forces_normal_human_approval(
    confirm_config, monkeypatch, choice, expected
):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

    result, seen = _run_in_gateway("aws s3 ls", choice)

    assert result["approved"] is expected
    if expected:
        assert result.get("user_approved") is True
    assert len(seen) == 1
    assert seen[0]["allow_permanent"] is False
    assert seen[0]["choices"] == ["once", "deny"]
    assert seen[0]["one_shot_only"] is True


@pytest.mark.parametrize("choice", ["session", "always"])
def test_gateway_session_and_always_choices_never_persist(
    confirm_config, monkeypatch, choice
):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

    first, first_seen = _run_in_gateway("aws s3 ls", choice)
    second, second_seen = _run_in_gateway("aws s3 ls", "deny")

    assert first["approved"] is True
    assert second["approved"] is False
    assert len(first_seen) == len(second_seen) == 1
    assert approval.is_approved(SESSION_KEY, CONFIRM_KEY) is False
    assert CONFIRM_KEY not in approval._permanent_approved


def test_gateway_bulk_approval_cannot_approve_one_shot_entries():
    confirm_entry = approval._ApprovalEntry({"one_shot_only": True})
    ordinary_entry = approval._ApprovalEntry({"one_shot_only": False})
    with approval._lock:
        approval._gateway_queues[SESSION_KEY] = [confirm_entry, ordinary_entry]

    resolved = approval.resolve_gateway_approval(
        SESSION_KEY, "once", resolve_all=True
    )

    assert resolved == 1
    assert ordinary_entry.event.is_set()
    assert ordinary_entry.result == "once"
    assert confirm_entry.event.is_set() is False
    with approval._lock:
        assert approval._gateway_queues[SESSION_KEY] == [confirm_entry]

    assert approval.resolve_gateway_approval(SESSION_KEY, "once") == 1
    assert confirm_entry.event.is_set()


@pytest.mark.parametrize("choice", ["session", "always"])
def test_interactive_session_and_always_choices_never_persist(
    confirm_config, monkeypatch, choice
):
    _interactive(monkeypatch)
    token = approval.set_current_session_key(SESSION_KEY)
    calls = 0

    def choose_scope(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return choice if calls == 1 else "deny"

    try:
        first = approval.check_all_command_guards(
            "aws s3 ls", "local", approval_callback=choose_scope
        )
        second = approval.check_all_command_guards(
            "aws s3 ls", "local", approval_callback=choose_scope
        )
    finally:
        approval.reset_current_session_key(token)

    assert first["approved"] is True
    assert second["approved"] is False
    assert calls == 2
    assert approval.is_approved(SESSION_KEY, CONFIRM_KEY) is False


def test_cron_confirm_fails_closed_even_in_approve_mode(
    confirm_config, monkeypatch
):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    result = approval.check_all_command_guards("aws s3 ls", "local")

    assert result["approved"] is False
    assert "BLOCKED" in result["message"]
    assert "cron" in result["message"].lower()
    assert "aws" in result["message"].lower()
    assert "cron_mode" in result["message"]


def test_cron_confirm_ignores_inherited_interactive_flag(
    confirm_config, monkeypatch
):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    result = approval.check_all_command_guards(
        "aws s3 ls",
        "local",
        approval_callback=lambda *_args, **_kwargs: pytest.fail(
            "cron must not open a live approval prompt"
        ),
    )

    assert result["approved"] is False
    assert "cron" in result["message"].lower()


def test_nonmatching_dangerous_cron_command_keeps_cron_mode_behavior(
    confirm_config, monkeypatch
):
    confirm_config["mode"] = "manual"
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    result = approval.check_all_command_guards("rm -rf /tmp/stuff", "local")

    assert result == {"approved": True, "message": None}


def test_headless_confirm_fails_closed(confirm_config):
    result = approval.check_all_command_guards("aws s3 ls", "local")

    assert result["approved"] is False
    assert "BLOCKED" in result["message"]
    assert "no interactive" in result["message"].lower()


def test_exec_ask_pending_preserves_one_shot_policy(confirm_config, monkeypatch):
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    result = approval.check_all_command_guards("aws s3 ls", "local")

    assert result["approved"] is False
    assert result["status"] == "approval_required"
    assert result["allow_permanent"] is False
    assert result["one_shot_only"] is True
    assert result["choices"] == ["once", "deny"]


def test_isolated_docker_may_bypass_but_host_bound_docker_is_protected(
    confirm_config
):
    isolated = approval.check_all_command_guards("aws s3 ls", "docker")
    host_bound = approval.check_all_command_guards(
        "aws s3 ls", "docker", has_host_access=True
    )

    assert isolated == {"approved": True, "message": None}
    assert host_bound["approved"] is False


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.run(['aws', 's3', 'ls'])",
        "import subprocess\nsubprocess.call(('/usr/local/bin/AWS', 's3', 'ls'))",
        "import subprocess\nsubprocess.check_call(['./aws', 's3', 'ls'])",
        "import subprocess\nsubprocess.check_output(['../bin/aws', 's3', 'ls'])",
        "import subprocess\nsubprocess.Popen(['aws', 's3', 'ls'])",
        "import subprocess\nsubprocess.run(args=['aws', 's3', 'ls'])",
        "import subprocess\nsubprocess.run(['sudo', 'aws', 's3', 'ls'])",
        (
            "import subprocess\n"
            "subprocess.run(['env', 'AWS_PROFILE=prod', 'aws', 's3', 'ls'])"
        ),
        (
            "import subprocess\n"
            "subprocess.run(['env', '-S', 'AWS_PROFILE=prod aws s3 ls'])"
        ),
        (
            "import subprocess\n"
            "subprocess.run(['echo', 'safe'], executable='aws')"
        ),
        "import subprocess\nsubprocess.run('sudo aws s3 ls', shell=True)",
        "import os\nos.system('env AWS_PROFILE=prod aws s3 ls')",
        "import os\nos.system(command='aws s3 ls')",
    ],
)
def test_execute_code_literal_subprocess_and_os_system_calls_are_confirmed(
    confirm_config, code
):
    result = approval.check_execute_code_guard(code, "local")

    assert result["approved"] is False
    assert "BLOCKED" in result["message"]
    assert "aws" in result["message"].lower()


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess as sp\nsp.run(['aws', 's3', 'ls'])",
        "from subprocess import run as r\nr(['aws', 's3', 'ls'])",
        "from subprocess import Popen as start\nstart(('aws', 's3', 'ls'))",
        "import os as operating_system\noperating_system.system('aws s3 ls')",
        "from os import system as invoke\ninvoke('aws s3 ls')",
        (
            "import subprocess\n"
            "cmd = ['aws', 's3', 'ls']\n"
            "subprocess.run(cmd)"
        ),
        (
            "import subprocess as sp\n"
            "cmd = ('/usr/local/bin/AWS', 's3', 'ls')\n"
            "sp.check_output(cmd)"
        ),
        "import os\ncmd = 'aws s3 ls'\nos.system(cmd)",
        (
            "import subprocess as sp\n"
            "cmd = 'sudo aws s3 ls'\n"
            "sp.run(cmd, shell=True)"
        ),
        (
            "import subprocess\n"
            "cmd = ['aws', 's3', 'ls']\n"
            "subprocess.run(cmd)\n"
            "cmd = ['echo', 'safe']"
        ),
        (
            "import subprocess\n"
            "if False:\n"
            "    subprocess = user_runner\n"
            "subprocess.run(['aws', 's3', 'ls'])"
        ),
        (
            "import subprocess\n"
            "for _ in []:\n"
            "    subprocess = user_runner\n"
            "subprocess.run(['aws', 's3', 'ls'])"
        ),
        (
            "import subprocess\n"
            "try:\n"
            "    pass\n"
            "except Exception:\n"
            "    subprocess = user_runner\n"
            "subprocess.run(['aws', 's3', 'ls'])"
        ),
        (
            "import subprocess\n"
            "match 0:\n"
            "    case 1:\n"
            "        subprocess = user_runner\n"
            "subprocess.run(['aws', 's3', 'ls'])"
        ),
    ],
)
def test_execute_code_import_aliases_and_assigned_literals_are_confirmed(
    confirm_config, code
):
    result = approval.check_execute_code_guard(code, "local")

    assert result["approved"] is False
    assert result.get("executable_confirm") is True
    assert "BLOCKED" in result["message"]
    assert "aws" in result["message"].lower()


@pytest.mark.parametrize(
    "code",
    [
        "'aws s3 ls'",
        "# subprocess.run(['aws', 's3', 'ls'])\nprint('safe')",
        "import subprocess\nimport boto3\nboto3.client('s3')",
        "import subprocess\nsubprocess.run(['echo', 'aws'])",
        "import subprocess\nsubprocess.run(['sudo', 'echo', 'aws'])",
        (
            "import subprocess\n"
            "subprocess.run(['env', 'AWS_NOTE=aws', 'echo', 'safe'])"
        ),
        (
            "import subprocess\n"
            "subprocess.run(['echo', 'safe'], executable='aws-helper')"
        ),
        (
            "import subprocess\n"
            "command = 'aws'\n"
            "subprocess.run(['sudo', command])"
        ),
        "import subprocess\nsubprocess.run('aws s3 ls')",
        "from hermes_tools import terminal\nterminal('aws s3 ls')",
        "import subprocess\nsubprocess.run(['aws'",
    ],
)
def test_execute_code_false_positives_and_syntax_errors_are_ignored(
    confirm_config, code
):
    result = approval.check_execute_code_guard(code, "local")

    assert result == {"approved": True, "message": None}


@pytest.mark.parametrize(
    "code",
    [
        "subprocess.run(['aws', 's3', 'ls'])",
        "os.system('aws s3 ls')",
        (
            "import user_subprocess as subprocess\n"
            "subprocess.run(['aws', 's3', 'ls'])"
        ),
        "from helpers import run as r\nr(['aws', 's3', 'ls'])",
        "import user_os as os\nos.system('aws s3 ls')",
        (
            "def run(args):\n"
            "    return args\n"
            "run(['aws', 's3', 'ls'])"
        ),
        "print('aws')",
        (
            "import subprocess as sp\n"
            "sp = user_runner\n"
            "sp.run(['aws', 's3', 'ls'])"
        ),
        (
            "import subprocess\n"
            "def helper(subprocess):\n"
            "    subprocess.run(['sudo', 'aws', 's3', 'ls'], executable='aws')"
        ),
        (
            "import subprocess\n"
            "def helper():\n"
            "    subprocess = user_runner\n"
            "    subprocess.run(['aws', 's3', 'ls'])"
        ),
        (
            "from subprocess import run as r\n"
            "r = user_run\n"
            "r(['aws', 's3', 'ls'])"
        ),
        (
            "import subprocess\n"
            "cmd = ['aws', 's3', 'ls']\n"
            "cmd = make_command()\n"
            "subprocess.run(cmd)"
        ),
        (
            "import os\n"
            "cmd = 'aws s3 ls'\n"
            "cmd = 'echo safe'\n"
            "os.system(cmd)"
        ),
        (
            "import subprocess\n"
            "cmd = ['aws'] if use_aws else ['echo']\n"
            "subprocess.run(cmd)"
        ),
    ],
)
def test_execute_code_requires_real_imports_and_respects_shadowing(
    confirm_config, code
):
    result = approval.check_execute_code_guard(code, "local")

    assert result == {"approved": True, "message": None}


def test_execute_code_gateway_approval_is_one_shot(
    confirm_config, monkeypatch
):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    code = "import subprocess\nsubprocess.run(['aws', 's3', 'ls'])"

    first, first_seen = _run_code_in_gateway(code, "always")
    second, second_seen = _run_code_in_gateway(code, "deny")

    assert first["approved"] is True
    assert second["approved"] is False
    assert len(first_seen) == len(second_seen) == 1
    assert first_seen[0]["choices"] == ["once", "deny"]
    assert approval.is_approved(SESSION_KEY, CONFIRM_KEY) is False
    assert approval.is_approved(SESSION_KEY, "execute_code") is False


def test_execute_code_confirm_beats_process_yolo(
    confirm_config, monkeypatch
):
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    code = "import os\nos.system('aws s3 ls')"

    result = approval.check_execute_code_guard(code, "local")

    assert result["approved"] is False


def test_execute_code_confirm_fails_closed_in_cron_approve(
    confirm_config, monkeypatch
):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    code = "import subprocess\nsubprocess.run(['aws', 's3', 'ls'])"

    result = approval.check_execute_code_guard(code, "local")

    assert result["approved"] is False
    assert "BLOCKED" in result["message"]
    assert "cron" in result["message"].lower()


def test_nonmatching_execute_code_keeps_cron_approve_behavior(
    confirm_config, monkeypatch
):
    confirm_config["mode"] = "manual"
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    result = approval.check_execute_code_guard("print('safe')", "local")

    assert result == {"approved": True, "message": None}


def test_host_bound_execute_code_is_protected_but_isolated_docker_may_bypass(
    confirm_config
):
    code = "import os\nos.system('aws s3 ls')"

    isolated = approval.check_execute_code_guard(code, "docker")
    host_bound = approval.check_execute_code_guard(
        code, "docker", has_host_access=True
    )

    assert isolated == {"approved": True, "message": None}
    assert host_bound["approved"] is False
