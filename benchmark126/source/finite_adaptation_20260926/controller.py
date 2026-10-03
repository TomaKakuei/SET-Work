"""Damping from the actual intervention displacement and existing curvature."""
import math


def intervention_damping(displacement, gradient, curvature, before, after,
                         damping_before, config):
    # This is the registered outer quadratic model, including its damping.
    # Never reuse the old direction's scale or predicted reduction.
    directional = float(gradient @ displacement)
    quadratic = float(displacement @ (curvature @ displacement))
    predicted = -(directional + .5 * quadratic)
    actual = float(before.mean() - after.mean())
    valid = math.isfinite(predicted) and math.isfinite(actual) and predicted > 0
    ratio = actual / predicted if valid else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    if ratio is None or ratio < .25:
        damping = min(config.damping_max, damping_before * 3.)
        reason = 'invalid_or_poor_prediction'
    elif ratio > .75:
        damping = max(config.damping_min, damping_before * .3)
        reason = 'accurate_full_intervention'
    else:
        damping = damping_before
        reason = 'retain'
    return damping, dict(predicted=predicted, actual=actual, ratio=ratio,
                         reason=reason, damping_after=damping,
                         prediction_source='existing_mean_damped_curvature')
