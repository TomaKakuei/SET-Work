# RNC finite-trajectory trust and expanded LM proxy review

Status: completed and independently replayed on 2026-09-16. No background
process remains. All optimizer routes use five outer steps. There was no
training, checkpoint change, task-conditioned routing, or active solver change.

## Result

The task-independent finite-trajectory trust rule turned the unstable direct
RNC2 adaptation into a useful CSN candidate. Against the frozen straight CSN,
`csn_rnc2_trust` records 2 wins, 18 ties, and 1 loss on the selected 21 cases.
Against the registered original LM it records 21 wins and no ties or losses;
against registered PCG16 it also records 21 wins and no ties or losses.

The material change is on multiframe:

| Split/case | Straight CSN | Trusted RNC2 CSN | Favorable change |
|---|---:|---:|---:|
| confirmation `case_0542` | 13.5698279 px | 12.3083967 px | +1.2614312 px |
| development `case_0543` | 0.163698347 px | 0.163656220 px | +0.000042127 px |
| development `case_0540` | 2.57859846 px | 2.58950169 px | -0.01090323 px |

The other 18 cases tie the straight CSN under the preregistered score tolerance.
In particular, the direct untrusted RNC2 losses on lowlight, SIDD, and stereo
are all restored to the straight CSN endpoint. The remaining `case_0540`
change does not lose its direction-specific lead: trusted CSN is still
0.6820925 px better than original LM on that case.

Against the explicit-residual trusted LM proxy, trusted CSN records 20 wins and
1 loss. The loss is development SIDD `case_0810`, where the LM proxy is better
by 0.00143433 dB. Against the per-case best of original LM, explicit-residual
trusted LM, and matched-curvature trusted LM, trusted CSN records 12 wins,
4 ties, and 5 losses. This is the appropriate detailed LM envelope for this
screen; it does not erase the 21/21 result against the registered original LM.

## What changed

The first screen crossed the complete registered CSN tangent and an explicit
LM tangent with straight and order-two RNC finite updates. The straight CSN
route reproduced the frozen incumbent with zero relative parameter error. The
untrusted curve was 4/9/8 against the straight CSN. Its failures coincided with
raw acceleration ratios ranging from tens through 1.55e9, while linear residual
controls produced effectively zero acceleration.

The follow-up adds one uniform finite-trajectory trust rule:

1. For `theta(t)=theta+t*v+t^2*a/2`, scale `a` so that
   `||a/2|| <= ||v||` at `t=1`.
2. At every curve time, use the curved endpoint only when both registered view
   costs are no larger than their straight-endpoint values. Otherwise that
   outer step uses the straight endpoint.

The rule reads no truth metric, task label, or condition. It selected the curve
on 16 of 105 CSN outer steps and used the straight fallback on 89. All 105 CSN
outer steps were accepted. The confirmation multiframe gain comes from two
curve selections; the other steps retain the straight path.

## Expanded LM proxy result

The comparison now brackets LM in three ways:

- `lm_original`: the registered historical project LM control.
- `lm_proxy_*`: explicit residual Gauss-Newton with Moré diagonal damping.
- `matched_lm_*`: direct solve in the exact Stage5 curvature used to construct
  the CSN proposal.

The trusted curve adds no reliable gain to the explicit residual proxy:
0 wins, 20 ties, and 1 loss against its straight version. The matched-curvature
LM behaves differently: trusted RNC2 is 2 wins, 19 ties, and 0 losses against
matched-curvature straight LM. It improves development `case_0540` by
0.0132770 px and confirmation `case_0542` by 0.00939303 px. The trusted matched
LM is 10 wins, 11 ties, and 0 losses against original LM.

The confirmation multiframe improvement is therefore much larger when the
trusted curve follows the CSN tangent: 1.2614312 px over straight CSN versus
0.00939303 px for the matched direct LM curve over matched straight LM. The
result supports a specific interaction between the CSN direction and finite
trajectory geometry rather than a generic benefit from adding an RNC2 term.

## Verification

The first screen generated 105 endpoints. Its independent replay checked 210
score rows with maximum score error 0 and confirmed exact straight-CSN
parameter reproduction. The trust screen generated 63 additional endpoints.
Its independent replay checked all 231 combined score rows with maximum score
error 0. Frozen prediction hashes are recorded in both run directories.

The implementation remains research-only Python orchestration around the
existing float64 C-backed task interfaces. `stage5.py`, the active policy, the
shared 49,096-parameter checkpoint, and the deployed native core are unchanged.
The candidate is ready for a native C implementation of the trust envelope and
a focused confirmation before changing the active solver.
