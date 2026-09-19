from io import BytesIO
from zipfile import ZipFile

from feature_extractor.utils.logger import LogConfig


def test_download_logs_404_when_no_log_files(client, monkeypatch, tmp_path):
    monkeypatch.setattr(LogConfig, "LOG_DIR", str(tmp_path))
    response = client.get("/logs")
    assert response.status_code == 404


def test_download_logs_returns_zip(client, monkeypatch, tmp_path):
    (tmp_path / "service.log").write_bytes(b"line one\nline two\n")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    monkeypatch.setattr(LogConfig, "LOG_DIR", str(tmp_path))

    response = client.get("/logs")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"].endswith('Z.zip"')

    with ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["service.log"]
        assert archive.read("service.log") == b"line one\nline two\n"
