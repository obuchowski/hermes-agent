"""Regression coverage for chat startup and compressed-session resolution."""

import pytest


def test_fresh_chat_uses_configured_tui(monkeypatch, tmp_path):
    import hermes_cli.main as main
    from hermes_cli._parser import build_top_level_parser

    (tmp_path / "config.yaml").write_text("display:\n  interface: tui\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_TUI", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(main, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(main, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap", lambda **kw: None)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_consent.offer_consent_before_chat",
        lambda args: None,
    )
    launched = []

    def launch(resume, **kwargs):
        launched.append(resume)
        raise SystemExit(0)

    monkeypatch.setattr(main, "_launch_tui", launch)
    parser, _, _ = build_top_level_parser()
    with pytest.raises(SystemExit) as exc:
        main.cmd_chat(parser.parse_args([]))
    assert exc.value.code == 0
    assert launched == [None]


@pytest.mark.parametrize("reference", ["root", "Compressed work"])
def test_resume_follows_compression_chain(reference, monkeypatch, tmp_path):
    from hermes_cli.main import _resolve_session_by_name_or_id
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB()
    try:
        db.create_session("root", source="cli")
        db.set_session_title("root", "Compressed work")
        db.end_session("root", end_reason="compression")
        db.create_session("middle", source="cli", parent_session_id="root")
        db.end_session("middle", end_reason="compression")
        db.create_session("tip", source="cli", parent_session_id="middle")
    finally:
        db.close()

    assert _resolve_session_by_name_or_id(reference) == "tip"
    assert _resolve_session_by_name_or_id("tip") == "tip"
    assert _resolve_session_by_name_or_id("missing") is None
