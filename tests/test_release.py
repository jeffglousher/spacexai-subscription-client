"""Tests for the offline release and distribution contract."""

import io
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from script.check_release import (
    main,
    validate_artifacts,
    validate_project,
    validate_release_source,
)

VERSION = "0.1.0"
COMMIT = "a" * 40
DISTRIBUTION = f"spacexai_subscription_client-{VERSION}"
METADATA = f"{DISTRIBUTION}.dist-info/METADATA"
WHEEL_FORMAT = f"{DISTRIBUTION}.dist-info/WHEEL"
WHEEL_RECORD = f"{DISTRIBUTION}.dist-info/RECORD"
TYPING_MARKER = "spacexai_subscription_client/py.typed"
WHEEL_LICENSE = f"{DISTRIBUTION}.dist-info/licenses/LICENSE"


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create the minimum expected release source tree."""
    files = {
        "pyproject.toml": (
            '[project]\nname = "spacexai-subscription-client"\nversion = "0.1.0"\n'
        ),
        "CHANGELOG.md": "# Changelog\n\n## 0.1.0\n\n- Initial release.\n",
        "README.md": "Package instructions\n",
        "RELEASING.md": "Release instructions\n",
        "LICENSE": "Apache License, Version 2.0\n",
        "uv.lock": "version = 1\n",
        ".github/workflows/ci.yml": "name: CI\n",
        ".github/workflows/release.yml": "name: Release\n",
        "src/spacexai_subscription_client/__init__.py": '"""Client."""\n',
        f"src/{TYPING_MARKER}": "",
        "tests/test_client.py": '"""Tests."""\n',
        "script/check_release.py": '"""Release checks."""\n',
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def artifact_files(project_dir: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Prepare matching wheel and source distribution contents."""
    source = {
        path.relative_to(project_dir).as_posix(): path.read_bytes()
        for path in project_dir.rglob("*")
        if path.is_file()
    }
    wheel = {
        name.removeprefix("src/"): content
        for name, content in source.items()
        if name.startswith("src/")
    }
    wheel[WHEEL_LICENSE] = source["LICENSE"]
    wheel[METADATA] = (
        b"Metadata-Version: 2.4\n"
        b"Name: spacexai-subscription-client\n"
        b"Version: 0.1.0\n"
        b"License-Expression: Apache-2.0\n"
    )
    wheel[WHEEL_FORMAT] = b"Wheel-Version: 1.0\nTag: py3-none-any\n"
    wheel[WHEEL_RECORD] = b""
    return wheel, source


def _write_artifacts(
    root: Path, files: tuple[dict[str, bytes], dict[str, bytes]]
) -> Path:
    """Write fixture archives without extracting files."""
    dist = root / "dist"
    dist.mkdir()
    wheel, source = files
    with ZipFile(dist / f"{DISTRIBUTION}-py3-none-any.whl", "w") as archive:
        for name, content in wheel.items():
            archive.writestr(name, content)
    with tarfile.open(dist / f"{DISTRIBUTION}.tar.gz", "w:gz") as archive:
        for name, content in source.items():
            member = tarfile.TarInfo(f"{DISTRIBUTION}/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return dist


def test_valid_project(project_dir: Path) -> None:
    """Require release notes for the declared project version."""
    assert validate_project(project_dir) == VERSION


@pytest.mark.parametrize(
    "notes",
    [
        pytest.param("## 0.2.0\n\n- Another release.\n", id="wrong-version"),
        pytest.param("## 0.1.0\n\n", id="empty-notes"),
        pytest.param("## 0.1.0\n\n## 0.0.1\n\n- Old release.\n", id="empty-section"),
    ],
)
def test_missing_release_notes(project_dir: Path, notes: str) -> None:
    """Do not mistake an older release's notes for the requested release."""
    (project_dir / "CHANGELOG.md").write_text(notes, encoding="utf-8")
    with pytest.raises(ValueError, match="Missing release notes"):
        validate_project(project_dir)


def test_wrong_project(project_dir: Path) -> None:
    """Reject the former distribution name."""
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "spacexai-client"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unexpected project name"):
        validate_project(project_dir)


def test_verified_release_source() -> None:
    """Allow the version tag only on the current main commit."""
    validate_release_source(VERSION, "v0.1.0", COMMIT, COMMIT)


@pytest.mark.parametrize(
    ("tag", "commit", "main_commit", "message"),
    [
        pytest.param("v0.2.0", COMMIT, COMMIT, "tag", id="version-mismatch"),
        pytest.param("v0.1.0", "b" * 40, COMMIT, "current main", id="unmerged-commit"),
        pytest.param("v0.1.0", COMMIT, "b" * 40, "current main", id="main-advanced"),
        pytest.param("v0.1.0", "", "", "current main", id="missing-commits"),
    ],
)
def test_invalid_release_source(
    tag: str, commit: str, main_commit: str, message: str
) -> None:
    """Reject stale or unrelated release sources before building."""
    with pytest.raises(ValueError, match=message):
        validate_release_source(VERSION, tag, commit, main_commit)


def test_valid_artifacts(
    project_dir: Path, artifact_files: tuple[dict[str, bytes], dict[str, bytes]]
) -> None:
    """Accept both artifacts with the expected source contents."""
    dist = _write_artifacts(project_dir, artifact_files)
    (dist / ".gitignore").write_text("*", encoding="utf-8")
    validate_artifacts(project_dir, dist, VERSION)


@pytest.mark.parametrize(
    ("index", "member", "message"),
    [
        pytest.param(0, TYPING_MARKER, "Missing wheel", id="typing-marker"),
        pytest.param(0, WHEEL_LICENSE, "Missing wheel", id="wheel-license"),
        pytest.param(0, METADATA, "Missing wheel", id="wheel-metadata"),
        pytest.param(0, WHEEL_FORMAT, "Missing wheel", id="wheel-format"),
        pytest.param(0, WHEEL_RECORD, "Missing wheel", id="wheel-record"),
        pytest.param(1, "tests/test_client.py", "Missing source", id="tests"),
        pytest.param(
            1, ".github/workflows/release.yml", "Missing source", id="workflow"
        ),
        pytest.param(1, "script/check_release.py", "Missing source", id="checker"),
        pytest.param(1, "README.md", "Missing source", id="readme"),
    ],
)
def test_missing_artifact_content(
    project_dir: Path,
    artifact_files: tuple[dict[str, bytes], dict[str, bytes]],
    index: int,
    member: str,
    message: str,
) -> None:
    """Imports alone cannot prove these required distribution files exist."""
    del artifact_files[index][member]
    dist = _write_artifacts(project_dir, artifact_files)
    with pytest.raises(ValueError, match=message):
        validate_artifacts(project_dir, dist, VERSION)


@pytest.mark.parametrize(
    ("index", "member"),
    [
        pytest.param(0, "spacexai_subscription_client/__init__.py", id="wheel-source"),
        pytest.param(1, "pyproject.toml", id="sdist-source"),
    ],
)
def test_changed_artifact_content(
    project_dir: Path,
    artifact_files: tuple[dict[str, bytes], dict[str, bytes]],
    index: int,
    member: str,
) -> None:
    """Do not publish distributions that differ from the verified checkout."""
    artifact_files[index][member] = b"Unexpected replacement\n"
    dist = _write_artifacts(project_dir, artifact_files)
    with pytest.raises(ValueError, match="differs from source"):
        validate_artifacts(project_dir, dist, VERSION)


@pytest.mark.parametrize(
    "member",
    [
        pytest.param("unexpected.pth", id="startup-path"),
        pytest.param(
            f"{DISTRIBUTION}.data/purelib/unexpected.py", id="relocated-module"
        ),
        pytest.param("spacexai_subscription_client/unexpected.py", id="package-module"),
        pytest.param("unexpected/__init__.py", id="extra-package"),
    ],
)
def test_unexpected_wheel_member(
    project_dir: Path,
    artifact_files: tuple[dict[str, bytes], dict[str, bytes]],
    member: str,
) -> None:
    """Reject installable files absent from the reviewed package source."""
    artifact_files[0][member] = b"# Unexpected file\n"
    dist = _write_artifacts(project_dir, artifact_files)
    with pytest.raises(ValueError, match="Unexpected wheel files"):
        validate_artifacts(project_dir, dist, VERSION)


@pytest.mark.parametrize(
    "member",
    [
        pytest.param(METADATA, id="metadata"),
        pytest.param("spacexai_subscription_client/__init__.py", id="source"),
    ],
)
def test_duplicate_wheel_member(
    project_dir: Path,
    artifact_files: tuple[dict[str, bytes], dict[str, bytes]],
    member: str,
) -> None:
    """Reject duplicate archive names even when their bytes agree."""
    dist = _write_artifacts(project_dir, artifact_files)
    with (
        ZipFile(dist / f"{DISTRIBUTION}-py3-none-any.whl", "a") as archive,
        pytest.warns(UserWarning, match="Duplicate name"),
    ):
        archive.writestr(member, artifact_files[0][member])
    with pytest.raises(ValueError, match="Duplicate wheel members"):
        validate_artifacts(project_dir, dist, VERSION)


@pytest.mark.parametrize("field", ["Name", "Version", "License-Expression"])
def test_changed_wheel_metadata(
    project_dir: Path,
    artifact_files: tuple[dict[str, bytes], dict[str, bytes]],
    field: str,
) -> None:
    """Twine-valid metadata still has to describe the intended release."""
    artifact_files[0][METADATA] = artifact_files[0][METADATA].replace(
        f"{field}: ".encode(), f"{field}: unexpected-".encode()
    )
    dist = _write_artifacts(project_dir, artifact_files)
    with pytest.raises(ValueError, match="metadata does not match"):
        validate_artifacts(project_dir, dist, VERSION)


def test_unexpected_artifact(
    project_dir: Path, artifact_files: tuple[dict[str, bytes], dict[str, bytes]]
) -> None:
    """Reject extra artifacts instead of accidentally publishing another version."""
    dist = _write_artifacts(project_dir, artifact_files)
    (dist / "unexpected.whl").touch()
    with pytest.raises(ValueError, match="exactly one"):
        validate_artifacts(project_dir, dist, VERSION)


def test_cli_success(
    project_dir: Path, artifact_files: tuple[dict[str, bytes], dict[str, bytes]]
) -> None:
    """Exercise the same entry point used by CI and release jobs."""
    dist = _write_artifacts(project_dir, artifact_files)
    main(
        [
            "--root",
            str(project_dir),
            "--dist",
            str(dist),
            "--release-tag",
            "v0.1.0",
            "--release-commit",
            COMMIT,
            "--main-commit",
            COMMIT,
        ]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["--release-tag", "v0.1.0"], id="incomplete-provenance"),
        pytest.param(["--release-tag", ""], id="empty-provenance"),
        pytest.param(["--dist", "missing-directory"], id="missing-distributions"),
    ],
)
def test_cli_failure(project_dir: Path, arguments: list[str]) -> None:
    """CI receives a nonzero exit status when preflight cannot prove readiness."""
    with pytest.raises(SystemExit) as err:
        main(["--root", str(project_dir), *arguments])
    assert err.value.code == 2
