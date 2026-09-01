# Releasing spacexai-subscription-client

Releases are immutable and are published only by the public GitHub Actions workflow. Do not upload distributions with a local API token or with Twine from a workstation.

## One-time setup for 0.1.0

- [ ] Enable two-factor authentication on the maintainer's PyPI account and store the recovery codes securely.
- [ ] In the GitHub repository, create an environment named exactly `pypi`.
- [ ] Configure the `pypi` environment to allow deployments only from tags matching `v*`.
- [ ] Add the maintainer as a required reviewer for the `pypi` environment so every publication requires manual approval.
- [ ] In the PyPI account's **Publishing** settings, add a pending GitHub publisher with these exact values:
  - PyPI project name: `spacexai-subscription-client`
  - Owner: `jeffglousher`
  - Repository: `spacexai-subscription-client`
  - Workflow: `release.yml`
  - Environment: `pypi`
- [ ] Confirm that no long-lived PyPI API token is stored in GitHub Actions secrets.

A pending publisher creates the PyPI project during the first successful trusted publication. It does not reserve the project name before publication.

## Before publishing

- [ ] Review and understand every change in the release pull request.
- [ ] Confirm that `pyproject.toml` contains the intended version and that `CHANGELOG.md` has matching release notes.
- [ ] Confirm that the OAuth client identity and provider endpoints are still approved for this release.
- [ ] Confirm that the package public API has no API-key authentication mode or fallback.
- [ ] Confirm that GitHub Issues are enabled.
- [ ] Run `uv tree --no-dev` and review the complete locked runtime dependency tree.
- [ ] Confirm every direct and transitive runtime dependency uses an OSI-approved license compatible with Home Assistant's Apache-2.0 distribution.
- [ ] Confirm that pull-request CI passes on Python 3.12, 3.13, and 3.14.
- [ ] Build locally with `uv build --clear`.
- [ ] Check both artifacts with `uvx --from twine==7.0.0 twine check dist/*`.
- [ ] Inspect the wheel for `spacexai_subscription_client/py.typed` and the Apache-2.0 license.
- [ ] Inspect the source distribution for the package source, tests, README, changelog, license, and public workflows.
- [ ] Merge the release pull request to `main` without bypassing required checks.
- [ ] Confirm the merged `main` commit is the exact commit intended for the release.

## Publish

1. On GitHub, create a new release targeting the verified `main` commit.
2. Create the tag `v0.1.0` and title the release `spacexai-subscription-client 0.1.0`.
3. Use the `0.1.0` section of `CHANGELOG.md` as the release notes.
4. Publish the GitHub release.
5. Approve the `pypi` environment deployment when GitHub requests approval.
6. Wait for both release jobs to pass. The build job repeats linting, typing, tests, coverage, artifact building, and Twine validation before the isolated publish job requests an OIDC credential.

Do not rerun publication with `skip-existing`. PyPI releases cannot be replaced; investigate a failed workflow before retrying it.

## Verify after publishing

- [ ] Confirm `https://pypi.org/project/spacexai-subscription-client/0.1.0/` is available.
- [ ] Confirm PyPI shows version `0.1.0`, Python `>=3.12`, and license expression `Apache-2.0`.
- [ ] Confirm the PyPI project links include the source repository, issue tracker, and changelog.
- [ ] Confirm PyPI provides both `spacexai_subscription_client-0.1.0-py3-none-any.whl` and `spacexai_subscription_client-0.1.0.tar.gz`.
- [ ] Confirm PyPI shows trusted-publishing provenance and attestations for both files.
- [ ] Confirm the PyPI publisher is now listed as a normal trusted publisher rather than a pending publisher.
- [ ] In a clean environment, run `python -m pip install spacexai-subscription-client==0.1.0` and verify `python -c "from importlib.metadata import version; print(version('spacexai-subscription-client'))"` prints `0.1.0`.
- [ ] Import `SpaceXAISubscriptionClient` from `spacexai_subscription_client` in the clean environment.
- [ ] Confirm the Git tag `v0.1.0` points to the same commit used to build the published files.
- [ ] Record the GitHub release, workflow, PyPI project, and `main...v0.1.0` comparison links in the Home Assistant Core PR.

Only after these checks pass should the Home Assistant integration mark `dependency-transparency` as done and run its final hassfest validation.
