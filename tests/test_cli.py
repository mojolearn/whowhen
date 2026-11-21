"""Exercise the public terminal interface without fitting a model."""
import json

import pytest

from whowhen import cli


def test_table_command_reads_csv_and_writes_report(tmp_path, monkeypatch, capsys):
    table = tmp_path / "rows.csv"
    table.write_text("patient,hour,event,value\na,0,0,2\n")
    output = tmp_path / "report.json"
    calls = []

    class Report:
        def __str__(self):
            return "audit report"

        def to_dict(self):
            return {"checked": True}

        def headline(self):
            return {"overlap_gap": 0.03}

    def report(df, **kwargs):
        calls.append((df, kwargs))
        return Report()

    monkeypatch.setattr(cli, "report", report)
    monkeypatch.setattr("sys.argv", ["whowhen", str(table), "--unit", "patient",
                                   "--time", "hour", "--event", "event",
                                   "--horizon", "12", "--out", str(output),
                                   "--fail-if", "overlap_gap > 0.02"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert calls[0][0].to_dict("records") == [
        {"patient": "a", "hour": 0, "event": 0, "value": 2}]
    assert calls[0][1] == dict(unit="patient", time="hour", event="event",
                              importance=False, horizon=12.0, repeats=4,
                              random=False)
    assert json.loads(output.read_text()) == {"checked": True}
    captured = capsys.readouterr()
    assert "audit report" in captured.out
    assert "FAILED:" in captured.err


def test_demo_command(monkeypatch):
    from whowhen import demo
    calls = []
    monkeypatch.setattr(demo, "run_demo", lambda repeats: calls.append(repeats) or True)
    monkeypatch.setattr("sys.argv", ["whowhen", "demo", "--repeats", "2"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert calls == [2]


def test_help_shows_direct_table_command(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["whowhen", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--unit" in help_text
    assert "table" in help_text
    assert "whowhen demo" in help_text


def test_report_subcommand_is_removed(monkeypatch):
    monkeypatch.setattr("sys.argv", ["whowhen", "report", "rows.csv",
                                   "--unit", "patient", "--time", "hour"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_time_can_be_left_out(tmp_path, monkeypatch):
    table = tmp_path / "rows.csv"
    table.write_text("patient,value\na,2\n")
    calls = []

    class Report:
        def headline(self):
            return {}

    monkeypatch.setattr(cli, "report",
                        lambda df, **kw: calls.append(kw) or Report())
    monkeypatch.setattr("sys.argv", ["whowhen", str(table), "--unit",
                                     "patient"])
    cli.main()
    assert calls[0]["time"] is None


@pytest.mark.parametrize("flag, expected", [
    ([], False), (["--importance"], True),
    (["--importance", "a", "b"], ["a", "b"])])
def test_importance_flag(tmp_path, monkeypatch, flag, expected):
    table = tmp_path / "rows.csv"
    table.write_text("patient,a,b\nx,1,2\n")
    calls = []

    class Report:
        def headline(self):
            return {}

    monkeypatch.setattr(cli, "report",
                        lambda df, **kw: calls.append(kw) or Report())
    monkeypatch.setattr("sys.argv", ["whowhen", str(table), "--unit",
                                     "patient", *flag])
    cli.main()
    assert calls[0]["importance"] == expected

