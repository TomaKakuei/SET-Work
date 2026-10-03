# Original October stereo audit source

This code-only publication contains the two original research sources available in the downloaded October 3 checkpoint:

- [Cached stereo action-hull diagnostic](full126_training/stereo_action_hull_audit_v1/audit.py).
- [Independent saved-record auditor](full126_literature/stereo_action_hull_audit_v1/audit_saved_records.py).

Both files are preserved byte for byte. SOURCE_MANIFEST.json records their SHA-256 values and original archive identity. This publication contains no PDF, cached arrays, result tables, model candidates, or account metadata.

The code belongs to the separate cloud SET-Work checkout based on commit 8c61e288ee20f9f9ee126df58b800c5ca01b6bef. The existing original, quad, linear_fast and repair implementations from the earlier SETSUNET workspace are already published in benchmark126/source and code/setsunet_csn.

The diagnostic studies per-pixel allocation between the current disparity d, original proposal b, and local-label proposal z. It compares a numerical control and fixed barycenter with a TRAIN-truth-assisted teacher. The teacher is a privileged reachability diagnostic, not an inference router. The companion auditor recomputes cached vector arithmetic, scores, and guard conditions without importing an optimizer.

These historical entry points require the original stereo_local_labels_v1 observations, ordinary states, scorer arrays, frozen configuration/plans and recorded outputs. Those files are not included in this code-only publication. audit.py also uses Linux resource limits and alarm signals; it refuses an unaccounted rerun when STARTED.json exists. The two source files were syntax-checked and verified against original archive hashes; the full experiments were not rerun for publication.

This is a partial publication. The downloaded checkpoint does not contain the blur-inverse, lowlight-coupling, signed-homography, variance/CG, new learned-router, stereo half-rollout or SIDD-moment rollout implementations described in the Work Report. Their original cloud code/results capsules are needed to finish publishing those methods. The report's requests for historical T06 observations and external optimizer wrappers concern different material.
