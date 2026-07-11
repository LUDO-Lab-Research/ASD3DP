# GitHub Release Checklist — ASD3DP v0.4.1

## Repository metadata

- [ ] Repository name and short description are set.
- [ ] Zenodo DOI is used as the repository website.
- [ ] Topics are added.
- [ ] Social-preview image is uploaded.
- [ ] README placeholders are replaced.
- [ ] Maintainer contact is current.

## Citation and identifiers

- [ ] `CITATION.cff` contains all creators in citation order.
- [ ] ORCIDs and affiliations are verified.
- [ ] Zenodo DOI is inserted after reservation/publication.
- [ ] Associated paper DOI is linked when available.
- [ ] Version is `0.4.1` everywhere.
- [ ] Git tag is `v0.4.1`.

## Rights and privacy

- [ ] Final license is selected.
- [ ] `LICENSE` exists.
- [ ] Third-party files have provenance and rights records.
- [ ] VORON attribution is present.
- [ ] Audio has been screened for identifiable speech and private information.
- [ ] No endorsement by VORON Design is implied.

## Data validation

- [ ] 41 sessions are present.
- [ ] 164 session-channel WAV files are present.
- [ ] 28.90 h unique duration is reproduced.
- [ ] 115.58 channel-hours is reproduced.
- [ ] 4,779 normal and 4,216 anomaly clips are reproduced.
- [ ] 48 kHz and nominal 768 kb/s values are checked against WAV headers.
- [ ] Channel-position mapping is checked against `audio_channels.csv`.
- [ ] Every manifest path resolves on a case-sensitive system.
- [ ] Every clip maps to a source session, channel, and time interval.
- [ ] Four channels belonging to a session are time aligned.
- [ ] Checksums are generated and verified.

## Annotation validation

- [ ] Normal, soft, and hard definitions are consistent.
- [ ] Loose A, loose B, and loose A+B labels are verified.
- [ ] Extruder/nozzle labels distinguish protocol and physical interpretation.
- [ ] Toolhead-collision labels and boundaries are verified.
- [ ] Controlled protocols are distinguished from observed anomalies.
- [ ] Rule-based candidates remain distinct from verified ground truth.
- [ ] Slow/fast session labels are not presented as constant clip speeds.

## Evaluation

- [ ] Official partitions are session-disjoint.
- [ ] DCASE 2025 AE configuration is documented.
- [ ] Channel handling is documented.
- [ ] Feature extraction, normalization, seeds, AUC, and pAUC are reported.
- [ ] No unevaluated model is described as a completed baseline.

## Release

- [ ] Zenodo files and GitHub release files match the documented version.
- [ ] Release notes match the changelog.
- [ ] Clean download/extraction test passes.
- [ ] README links work.
- [ ] DOI badge works.
- [ ] Final release is not marked as a pre-release.
