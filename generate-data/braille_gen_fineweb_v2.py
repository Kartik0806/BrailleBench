"""Entry point for the braille dataset generator (kept so existing commands keep working).

    python src/braille_gen_fineweb_v2.py --texts_file texts.txt --output_dir data/train ...

The implementation lives in the braille_gen package (see its docstring); importing this
module re-exports the same namespace the former single-file generator had.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import braille_gen as _pkg  # noqa: E402

globals().update({k: v for k, v in vars(_pkg).items() if not k.startswith("__")})

if __name__ == "__main__":
    _pkg.cli.main()
