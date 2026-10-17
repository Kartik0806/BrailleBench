"""Optional third-party dependencies, with graceful fallbacks.

Every name is always bound (None when the package is missing) so other modules can import it;
code must still check the matching HAS_* flag before using it.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Optional deps — graceful fallback if unavailable
# ---------------------------------------------------------------------------
try:
    import noise as noise_lib  # pip install noise

    HAS_NOISE = True
except ImportError:
    noise_lib = None
    HAS_NOISE = False

try:
    from scipy.ndimage import map_coordinates, gaussian_filter

    HAS_SCIPY = True
except ImportError:
    map_coordinates = gaussian_filter = None
    HAS_SCIPY = False

try:
    import louis  # liblouis Python bindings; preferred for both UEB Grade-1 and Grade-2

    HAS_LIBLOUIS = True
except ImportError:
    louis = None
    HAS_LIBLOUIS = False
