"""STUB (roadmap Phase 14, section 28) -- NOT implemented in this pass.

Per section 52 ("don't overengineer") and the roadmap's own ordering, ML is
optional and comes only after the physics-based baseline (models/,
multimodal/) is working and evaluated (evaluation/) against real CARLA
data -- which this pass delivers. Building a training dataset before that
baseline exists and has been checked would be premature.

When this phase starts, this package should generate (X=history, Y=future
ground truth) pairs directly from evaluation/ground_truth.py's parsed
[GROUND TRUTH] blocks (per-actor world position/velocity history), NOT from
replayed M3 packets -- see section 28's two-variant note: train on clean
ground-truth history, but ALSO keep a variant built from recorded M4 input
packets (noisy, realistic M3 tracks) if the model needs to generalize to
M3's actual noise rather than clean simulator state. Split by SCENARIO
(section 29), never by randomly mixing frames from one trajectory across
train/test.

Ground truth is sufficient for POSITION/VELOCITY labels (x, y, vx, vy) but
carries no z, heading, size or class (evaluation/ground_truth.py's
docstring) -- if a dataset needs those, M1 must add a machine-readable
JSONL ground-truth log; this stub does not attempt to work around that gap
silently.
"""
