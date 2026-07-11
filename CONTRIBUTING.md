# Contributing to ASD3DP

Thank you for helping improve ASD3DP. Contributions may include documentation
corrections, manifest fixes, reproducibility improvements, baseline code, and
well-documented annotation corrections.

## Before opening an issue

Please include:

- dataset version;
- exact file path;
- session and channel identifier, when applicable;
- relevant time interval;
- expected behavior;
- observed behavior;
- a minimal reproducible example or validation command;
- whether the issue affects audio, metadata, annotation, code, or documentation.

Do not upload proprietary audio, private speech, or third-party data without
permission.

## Annotation corrections

An annotation correction should state:

1. the current label;
2. the proposed label;
3. the supporting audio/process evidence;
4. the relevant timestamps;
5. whether the change affects event-clean clip membership.

Corrections should preserve provenance. Rule-based candidates must not be
silently promoted to verified ground truth.

## Code contributions

- keep data paths configurable;
- do not assume a specific operating system;
- document the expected dataset version;
- use session-disjoint splitting by default;
- report random seeds and software versions;
- add a minimal test for manifest and path handling;
- avoid committing large audio files.

## Safety

Do not submit instructions that encourage unassessed reproduction of
toolhead-collision, nozzle-obstruction, or other equipment-damaging protocols.
Research descriptions should emphasize risk assessment and supervision.

## Pull requests

A pull request should contain:

- a clear purpose;
- changed files and expected effect;
- validation steps;
- compatibility notes for ASD3DP v0.4.1;
- updated documentation or changelog entry when needed.
