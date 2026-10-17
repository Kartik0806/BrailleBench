# source this to use the in-repo liblouis build (UEB g1/g2 tables) with any python
# override the location with LIBLOUIS_DIR=/path/to/liblouis-<version>
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LB="$(cd "${LIBLOUIS_DIR:-$_here/../../liblouis-3.38.0}" && pwd)"
export PYTHONPATH="$LB/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$LB/liblouis/.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LOUIS_TABLEPATH="$LB/tables"
unset _here
