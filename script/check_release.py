"""Validate release provenance, notes, and distribution contents without publishing."""

import argparse
import re
import tarfile
import tomllib
from collections.abc import Sequence
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

PROJECT_NAME = "spacexai-subscription-client"
PACKAGE_NAME = "spacexai_subscription_client"


def validate_project(root: Path) -> str:
    """Return the package version after checking its identity and release notes."""
    with (root / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)["project"]
    version = project["version"]
    if project["name"] != PROJECT_NAME or not isinstance(version, str):
        msg = "Unexpected project name or version"
        raise ValueError(msg)
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    section = re.search(
        rf"^## {re.escape(version)}[ \t]*\n(?P<notes>.*?)(?=^## |\Z)",
        changelog,
        flags=re.MULTILINE | re.DOTALL,
    )
    if section is None or not section["notes"].strip():
        msg = f"Missing release notes for {version}"
        raise ValueError(msg)
    return version


def validate_release_source(
    version: str, tag: str, release_commit: str, main_commit: str
) -> None:
    """Require a version tag on the current main commit, not attest human review."""
    if tag != f"v{version}":
        msg = "Release tag does not match the project version"
        raise ValueError(msg)
    if (
        re.fullmatch(r"[0-9a-f]{40}", release_commit) is None
        or release_commit != main_commit
    ):
        msg = "Release commit must be the current main commit"
        raise ValueError(msg)


def _source_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for directory in ("src", "tests", "script")
        for path in (root / directory).rglob("*.py")
    } | {
        "pyproject.toml",
        "README.md",
        "CHANGELOG.md",
        "RELEASING.md",
        "LICENSE",
        "uv.lock",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        f"src/{PACKAGE_NAME}/py.typed",
    }


def _check_content(root: Path, source: str, contents: bytes) -> None:
    if contents != (root / source).read_bytes():
        msg = f"Distribution content differs from source: {source}"
        raise ValueError(msg)


def _check_wheel_members(wheel: ZipFile, expected: set[str]) -> None:
    """Require each expected wheel member exactly once and no other files."""
    members = wheel.namelist()
    if len(members) != len(set(members)):
        msg = "Duplicate wheel members"
        raise ValueError(msg)
    if missing := expected - set(members):
        msg = f"Missing wheel files: {sorted(missing)}"
        raise ValueError(msg)
    if unexpected := set(members) - expected:
        msg = f"Unexpected wheel files: {sorted(unexpected)}"
        raise ValueError(msg)


def validate_artifacts(root: Path, dist: Path, version: str) -> None:
    """Check the two publishable artifacts against the checked-out source."""
    distribution = f"{PACKAGE_NAME}-{version}"
    wheel_name = f"{distribution}-py3-none-any.whl"
    sdist_name = f"{distribution}.tar.gz"
    if {path.name for path in dist.iterdir() if path.name != ".gitignore"} != {
        wheel_name,
        sdist_name,
    }:
        msg = "Expected exactly one version-matched wheel and source distribution"
        raise ValueError(msg)

    source_files = _source_files(root)
    metadata_path = f"{distribution}.dist-info/METADATA"
    with ZipFile(dist / wheel_name) as wheel:
        wheel_files = {
            source.removeprefix("src/"): source
            for source in source_files
            if source.startswith("src/")
        }
        wheel_files[f"{distribution}.dist-info/licenses/LICENSE"] = "LICENSE"
        expected = wheel_files.keys() | {
            metadata_path,
            f"{distribution}.dist-info/WHEEL",
            f"{distribution}.dist-info/RECORD",
        }
        _check_wheel_members(wheel, expected)
        for member, source in wheel_files.items():
            _check_content(root, source, wheel.read(member))
        metadata = Parser().parsestr(wheel.read(metadata_path).decode("utf-8"))
        if (
            metadata["Name"] != PROJECT_NAME
            or metadata["Version"] != version
            or metadata["License-Expression"] != "Apache-2.0"
        ):
            msg = "Wheel metadata does not match the release identity or license"
            raise ValueError(msg)

    with tarfile.open(dist / sdist_name) as sdist:
        missing = {f"{distribution}/{source}" for source in source_files} - set(
            sdist.getnames()
        )
        if missing:
            msg = f"Missing source distribution files: {sorted(missing)}"
            raise ValueError(msg)
        for source in source_files:
            source_member = sdist.getmember(f"{distribution}/{source}")
            if not source_member.isfile():
                msg = f"Source distribution member is not a file: {source}"
                raise ValueError(msg)
            contents = sdist.extractfile(source_member)
            if contents is None:
                msg = f"Unreadable source distribution file: {source}"
                raise ValueError(msg)
            with contents:
                _check_content(root, source, contents.read())


def main(argv: Sequence[str] | None = None) -> None:
    """Run offline checks, optionally with source provenance supplied by CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--release-tag")
    parser.add_argument("--release-commit")
    parser.add_argument("--main-commit")
    args = parser.parse_args(argv)
    try:
        version = validate_project(args.root)
        release_arguments = (
            args.release_tag,
            args.release_commit,
            args.main_commit,
        )
        if any(argument is not None for argument in release_arguments):
            if not all(release_arguments):
                parser.error("All three release provenance arguments are required")
            validate_release_source(version, *release_arguments)
        if args.dist is not None:
            validate_artifacts(args.root, args.dist, version)
    except (OSError, ValueError) as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
