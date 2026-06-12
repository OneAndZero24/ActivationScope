"""ActivationScope policy enumerations.

Three independently tunable policy knobs that govern every aspect of memory
and compute behaviour.  Integer-valued to pass trivially to the C++ backend.
"""


class _PolicyMeta(type):
    """Metaclass that makes policy classes iterable and subscriptable."""

    def __iter__(cls):
        return iter(cls._members_)

    def __class_getitem__(cls, item):
        return cls._members_[item]


class StoragePolicy(int, metaclass=_PolicyMeta):
    """Where tensor data lives after capture."""

    AUTO = 0    # Heuristic: < threshold → CPU, >= threshold → GPU
    CPU  = 1    # Transfer to host memory
    GPU  = 2    # Stay on original device
    DISK = 3    # Stream directly to disk; bypass in-memory accumulation

    _members_ = (AUTO, CPU, GPU, DISK)


class ReductionPolicy(int):
    """What gets kept vs. reduced across batches."""

    STORE_ALL  = 0  # Full tensor per batch appended
    STREAMING  = 1  # Per-batch reduction output replaces/accumulates in-place
    FINAL_ONLY = 2  # Last-batch activation overwrites previous


class CapturePolicy(int, metaclass=_PolicyMeta):
    """When and how often hooks fire."""

    EVERY    = 0  # Every forward fires hooks
    SAMPLE_N = 1  # Captures every Nth forward
    MAX_K    = 2  # Captures exactly K batches then stops

    _members_ = (EVERY, SAMPLE_N, MAX_K)


class CaptureMode(int, metaclass=_PolicyMeta):
    """Whether to clone captured tensors for independent storage.

    Controls the copy behaviour of both the native C++ tracker
    (``activationscope.tracker.ActivationScope``) and the pure‑Python
    fallback (``activationscope._naive.NaiveHookTracker``).
    """

    REFERENCE = 0  # detach() only — shares storage with autograd graph
    SNAPSHOT  = 1  # detach() + clone() — completely independent copy

    _members_ = (REFERENCE, SNAPSHOT)
