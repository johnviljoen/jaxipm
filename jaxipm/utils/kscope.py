"""Linear named-scope helper for the run-K fusion-cost profile.

`jax.named_scope` only attaches metadata to the traced ops (zero runtime cost);
`KScope` lets a long straight-line function switch scopes without re-indenting
whole blocks:  _ks = KScope(); _ks("derivs"); ...; _ks("kkt"); ...; _ks.close()
Scope names carry the prefix "K_" so the profiler parser can find them.
"""
import contextlib
import jax


class KScope:
    def __init__(self):
        self._st = None

    def __call__(self, name):
        self.close()
        self._st = contextlib.ExitStack()
        self._st.enter_context(jax.named_scope("K_" + name))

    def close(self):
        if self._st is not None:
            self._st.close()
            self._st = None
