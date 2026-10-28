# Changelog

All notable dataset and repository changes should be documented here.

## Repository documentation

- Links the published ASD3DP paper in the DCASE 2026 proceedings.
- Replaces the submission-status badge and pending paper citation with the
  official proceedings metadata.

## v0.4.3 public dataset release

- Adds or rebuilds derived annotation resources without changing the released
  audio or the soft/hard severity taxonomy.
- Keeps belt-tension and extruder cogging/no-extrusion conditions as
  `soft-anomaly`.
- Keeps toolhead collision as the only `hard-anomaly` fault mode in this
  release.

## Current documented dataset state

### Added

- 41 four-channel recording sessions.
- 164 session-channel WAV files.
- 28.99 h of unique printer operation.
- 115.97 channel-hours.
- 9,067 selected event-clean 10-second clips.
- Normal loaded-printing and no-filament motion-only conditions.
- Belt-tension, extruder-failure/nozzle-obstruction, and toolhead-collision
  anomaly conditions.
- Session and clip manifests.
- G-code and audio/G-code alignment resources.
- Telemetry synchronization.
- Phase and fault timelines.
- Initial DCASE 2025 Task 2 autoencoder baseline study.

### Known limitations

- One Voron 0.2-class printer build.
- One primary four-microphone layout.
- Some fault conditions are deliberately induced.
- Official cross-build domain-generalization split not yet available.
