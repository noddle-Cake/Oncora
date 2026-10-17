import subprocess

import pytest

from oncora.io import chembl_export

TOC = """\
;
; Archive created at 2026-05-05 10:30:00 BST
;
212; 1259 3148009 TABLE public component_sequences user
224; 1259 3148102 TABLE public target_dictionary user
300; 1259 3148500 TABLE public docs user
3012; 0 3148009 TABLE DATA public component_sequences user
3024; 0 3148102 TABLE DATA public target_dictionary user
3100; 0 3148500 TABLE DATA public docs user
4000; 1259 3149000 INDEX public idx_td_chembl user
4100; 2606 3149100 FK CONSTRAINT public fk_td user
"""


@pytest.fixture
def fake_pg_restore(monkeypatch):
    monkeypatch.setattr(chembl_export, "_pg_tool", lambda name: name)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=TOC, stderr=""),
    )


def test_restore_list_keeps_only_requested_tables(tmp_path, fake_pg_restore):
    dump = tmp_path / "x.dmp"
    tables = {"component_sequences": (), "target_dictionary": ()}
    lines = chembl_export.restore_list(dump, tables).read_text().splitlines()
    assert len(lines) == 4
    assert all(("TABLE public" in ln or "TABLE DATA public" in ln) for ln in lines)
    assert not any("docs" in ln or "INDEX" in ln or "FK" in ln for ln in lines)


def test_restore_list_fails_on_missing_table(tmp_path, fake_pg_restore):
    with pytest.raises(chembl_export.ChemblExportError, match="activities"):
        chembl_export.restore_list(tmp_path / "x.dmp", {"activities": ()})
