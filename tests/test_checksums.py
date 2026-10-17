import hashlib

import pytest

from oncora.io.checksums import RawDataError, verify_raw


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "a.csv"
    f.write_text("x,y\n1,2\n")
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  sub/a.csv\n")
    return tmp_path


def test_verify_ok(raw_dir):
    assert list(verify_raw(raw_dir / "SHA256SUMS")) == ["sub/a.csv"]


def test_tampered_file_detected(raw_dir):
    (raw_dir / "sub" / "a.csv").write_text("x,y\n1,3\n")
    with pytest.raises(RawDataError, match="checksum mismatch: sub/a.csv"):
        verify_raw(raw_dir / "SHA256SUMS")


def test_missing_file_detected(raw_dir):
    (raw_dir / "sub" / "a.csv").unlink()
    with pytest.raises(RawDataError, match="missing: sub/a.csv"):
        verify_raw(raw_dir / "SHA256SUMS")
