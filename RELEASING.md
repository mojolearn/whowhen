# Releasing

Two things cannot be undone: a PyPI version number can never be reused, even after the release is deleted, and a Zenodo record can never be deleted. Check everything before step 6.

1. Bump the version in `pyproject.toml`, `src/whowhen/__init__.py` and `CITATION.cff`.
2. Update `CHANGELOG.md`: rename the "(unreleased)" heading to the new version.
3. Run the tests: `python -m pytest -q`.
4. Commit and push, then wait for the test workflow to pass.
5. Tag the commit: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. Create the GitHub Release from that tag. This triggers:
   - Zenodo archiving and a new DOI, once the repository is switched on at zenodo.org/account/settings/github.
   - PyPI publishing by `.github/workflows/release.yml`, once the trusted publisher exists on pypi.org (see the comment at the top of that file).
7. After the first Zenodo DOI exists, add the concept DOI badge to `README.md` and a `doi:` field to `CITATION.cff`.
