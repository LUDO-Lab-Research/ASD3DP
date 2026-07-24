# GitHub Release Checklist — ASD3DP

## Repository metadata

- [ ] Repository name and short description are set.
- [ ] Zenodo DOI is used as the repository website.
- [ ] Topics are added.
- [ ] Social-preview image is uploaded.
- [x] README Zenodo placeholders are replaced.
- [ ] Maintainer contact is current.

## Citation and identifiers

- [x] `CITATION.cff` contains all creators in citation order.
- [ ] ORCIDs and affiliations are verified.
- [x] Zenodo DOI is inserted after publication.
- [ ] Associated paper DOI is linked when available.
- [ ] Release metadata is consistent across Zenodo and GitHub.

## Rights and privacy

- [x] Final licenses are selected (MIT code; CC BY 4.0 data).
- [x] `LICENSE` exists.
- [ ] Third-party files have provenance and rights records.
- [ ] VORON attribution is present.
- [ ] Audio has been screened for identifiable speech and private information.
- [ ] No endorsement by VORON Design is implied.

## Data validation

- [ ] 41 sessions are present.
- [ ] 164 session-channel WAV files are present.
- [ ] 28.99 h unique duration is reproduced.
- [ ] 115.97 channel-hours is reproduced.
- [ ] 4,766 normal and 4,301 anomaly clips are reproduced.
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
- [x] DOI badge works.
- [ ] Final release is not marked as a pre-release.
