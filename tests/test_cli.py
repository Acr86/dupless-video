"""CLI command tests (typer CliRunner). Kept light: --exact-only is hash-only (no torch/GPU/decode)."""
from __future__ import annotations

from typer.testing import CliRunner

from dupdetect.cli import app


def _lib(tmp_path):
    lib = tmp_path / "lib"; lib.mkdir()
    (lib / "a.mp4").write_bytes(b"x" * 100)
    (lib / "b.mp4").write_bytes(b"y" * 100)
    return lib


def test_scan_report_defaults_next_to_db_not_cwd(tmp_path, monkeypatch):
    """Regression (frozen app, WinError 5): the scan report must default NEXT TO THE DB, never a
    CWD-relative 'reports/'. The frozen app's CWD is the read-only install dir, so mkdir('reports')
    there raises PermissionError. Run with a CWD that is NOT the DB dir and assert where the report lands."""
    lib = _lib(tmp_path)
    dbdir = tmp_path / "store"; dbdir.mkdir()
    db = dbdir / "d.sqlite"
    cwd = tmp_path / "cwd"; cwd.mkdir()
    monkeypatch.chdir(cwd)                                  # CWD != db dir (mimics the install dir)
    res = CliRunner().invoke(app, ["scan", str(lib), "--db", str(db), "--exact-only"])
    assert res.exit_code == 0, res.output
    assert (dbdir / "reports" / "scan.json").exists()      # report co-located with the DB (writable)
    assert not (cwd / "reports").exists()                  # NOT created in the CWD (the old crash site)


def test_scan_explicit_out_is_respected(tmp_path, monkeypatch):
    """An explicit --out is honored and its parent dir is created, wherever it points."""
    lib = _lib(tmp_path)
    db = tmp_path / "d.sqlite"
    target = tmp_path / "custom" / "myreport.json"
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(app, ["scan", str(lib), "--db", str(db),
                                   "--exact-only", "--out", str(target)])
    assert res.exit_code == 0, res.output
    assert target.exists()
