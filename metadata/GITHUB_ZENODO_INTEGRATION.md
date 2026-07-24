# GitHub–Zenodo Integration Note

ASD3DP has two potentially different archival objects:

1. the **dataset deposit** containing audio and annotations; and
2. the **GitHub repository release** containing documentation and code.

Do not create duplicate or confusing DOI records unintentionally.

## Recommended arrangement

- Deposit the data files manually as a Zenodo **Dataset** record.
- Put the dataset DOI in `CITATION.cff`, the README badge, and the GitHub About
  website field.
- Archive the GitHub repository as a separate Zenodo software record only if a
  citable code snapshot is useful.
- Link the two records with related identifiers.

## CITATION.cff

GitHub recognizes a root-level `CITATION.cff`. Setting `type: dataset` makes the
GitHub citation prompt produce a dataset citation.

The creator fields and published Zenodo DOI are recorded in `CITATION.cff`.

## .zenodo.json

Zenodo can use a root-level `.zenodo.json` for GitHub-release metadata. If both
`.zenodo.json` and `CITATION.cff` are present, Zenodo uses `.zenodo.json` for
GitHub release archiving and ignores `CITATION.cff` for that ingestion.

This template therefore does **not** install a root `.zenodo.json` by default.
Add one only when you intentionally want the GitHub release archived as a
separate Zenodo record and need Zenodo-specific fields.
