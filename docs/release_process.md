# Release Process

This document describes MedAISure's release and versioning strategy, and provides a practical checklist for maintainers.

## Versioning Policy

We use Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`.

- `MAJOR`: incompatible API or behavior changes.
- `MINOR`: backwards-compatible functionality additions.
- `PATCH`: backwards-compatible bug fixes and documentation updates.

Pre-releases may use suffixes like `-alpha.1`, `-beta.1`, or `-rc.1`.

## Branching Model

- `main`: always green; accepts PRs via review + CI.
- `release/*`: temporary branches used for preparing a tagged release if needed (optional).
- `hotfix/*`: optional short-lived branches for emergency fixes to a released version.
- Feature branches follow `feature/<short-name>`.

## Tagging and Changelog

- Create annotated tags in the form `vX.Y.Z`.
- Maintain a human-readable `CHANGELOG.md` with sections for Added, Changed, Fixed, Removed.
  - For each release, summarize notable changes, breaking changes, and migration notes.
- For pre-releases, tag as `vX.Y.Z-rc.N` and state stability expectations.

## Release Checklist

1. Ensure `main` is green (CI passing).
2. Update version (if applicable):
   - `pyproject.toml` or packaging metadata (if publishing a package)
   - Any in-code version constants
3. Update `CHANGELOG.md` and docs as needed.
4. Run quality gates locally:
   - `pre-commit run --all-files`
   - `pytest` (with coverage)
   - Build docs: `mkdocs build`
5. Merge into `main` (or fast-forward) and create an annotated tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
6. Validate CI artifacts and release outputs.

## CI Release Steps (Guidelines)

Your CI should perform at least the following on tag push:

- Run test suite and linters
- Build documentation (`mkdocs build`) and publish site (e.g., GitHub Pages or artifact)
- Optionally: Build and publish Python package to an index (if applicable)
  - Environment secrets for tokens must be configured in CI
- Optionally: Attach build artifacts to GitHub Release

## Artifacts

- Documentation site (from `mkdocs build`)
- Python sdist/wheel (if distributing as a package)
- Optional: Benchmark result assets or sample datasets

## Backporting Policy

- Critical bug fixes can be backported to the most recent N minor versions (maintainers’ discretion).
- Process:
  1. Create a `backport/X.Y` branch from the target release tag
  2. Cherry-pick relevant fixes
  3. Bump `PATCH` version
  4. Repeat the release checklist and tag as `vX.Y.Z`

## Security and Responsible Disclosure

- For security issues, avoid public issues initially.
- Email maintainers at `security@medaisure.org` with details and reproduction steps.
- Maintainers will coordinate an expedited fix and release.

## Roles & Approvals

- At least one maintainer review is required to cut a release.
- For major releases, two approvals are recommended, with a migration guide in docs.

## Communication

- Announce releases via GitHub Releases and repository README badges.
- Update documentation navigation if new features introduce new pages.
