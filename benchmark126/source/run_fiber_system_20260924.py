"""Portable import surface for the frozen finite-model source.

The original research script is preserved as frozen_run_fiber_system_20260924.py.
Only its observation counter is needed by the released method modules.
"""
import numpy as np
import torch
import polynomial_basis as poly


class PhysicalAudit:
    def __init__(self, native):
        self.native = native
        self.events = []
        self.phase = 'solver'
        if hasattr(native, 'raw_jac'):
            self.attribute = 'raw_jac'
            self.original = native.raw_jac

            def call(x, v, jac):
                out = self.original(x, v, jac)
                self.events.append(dict(view=v, J=bool(jac), phase=self.phase))
                return out
        else:
            self.attribute = 'evaluate'
            self.original = native.evaluate

            def call(x, v, mode=2, *, jacobian=False):
                out = self.original(x, v, mode, jacobian=jacobian)
                self.events.append(dict(view=v, J=bool(mode > 0 or jacobian), phase=self.phase))
                return out
        setattr(native, self.attribute, call)

    def close(self):
        setattr(self.native, self.attribute, self.original)

    def counts(self):
        j = sum(x['J'] for x in self.events)
        f = len(self.events) - j
        assert j % 2 == f % 2 == 0
        return dict(actual_joint_J=j//2, joint_forward_only=f//2,
                    view_calls=len(self.events),
                    intervention_J=sum(x['J'] and x['phase'] == 'intervention'
                                       for x in self.events)//2,
                    intervention_forward=sum(not x['J'] and x['phase'] == 'intervention'
                                             for x in self.events)//2)
