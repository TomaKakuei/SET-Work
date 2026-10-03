"""Explicit opt-in shared initialization; original files remain versioned."""
from contextlib import contextmanager
from multiframe_scale_safe import decoder_normalization
from multiframe_initialization import initialize


@contextmanager
def prepared_problem(problem):
    if (problem.metadata or {}).get('family')!='hpatches_multiframe':
        yield problem,dict(changed=False,status='not_applicable',truth_read=False)
        return
    with decoder_normalization(problem):
        adapted,diagnostic=initialize(problem)
        yield adapted,diagnostic
