"""Build the two small C libraries with GCC, Clang, Zig, or TinyCC."""
from pathlib import Path
import argparse
import json
import os
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='cc', help='C compiler executable')
    frontend = parser.add_mutually_exclusive_group()
    frontend.add_argument('--zig', action='store_true', help='Use the Zig cc frontend')
    frontend.add_argument('--tcc', action='store_true', help='Use TinyCC (tested on Windows)')
    args = parser.parse_args()
    system = platform.system()
    suffix = '.dll' if system == 'Windows' else '.dylib' if system == 'Darwin' else '.so'
    environment = os.environ.copy()
    executable = str(Path(args.cc).resolve()) if Path(args.cc).is_file() else shutil.which(args.cc)
    if executable is None:
        parser.error(f'C compiler not found: {args.cc}')
    compiler = os.path.relpath(executable, ROOT)
    if args.zig:
        environment['ZIG_GLOBAL_CACHE_DIR'] = str(ROOT / 'build/zig-global-cache')
        environment['ZIG_LOCAL_CACHE_DIR'] = str(ROOT / 'build/zig-local-cache')
    for folder, stem in [('native_stage4', 'csn_small'), ('native_stage5', 'csn_factors')]:
        source = ROOT / 'code' / folder / (stem + '.c')
        target = source.with_suffix(suffix)
        command = [compiler] + (['cc'] if args.zig else [])
        if args.tcc:
            command += ['-B', str(Path(compiler).parent)]
        command += ['-dynamiclib' if system == 'Darwin' else '-shared']
        if not args.tcc:
            command += ['-O3', '-fno-fast-math', '-ffp-contract=off']
        if system != 'Windows' and not args.tcc:
            command += ['-fPIC']
        # Relative arguments also work with TinyCC under non-ASCII parent paths.
        command += ['-o', str(target.relative_to(ROOT)), str(source.relative_to(ROOT))]
        if system != 'Windows':
            command += ['-lm']
        subprocess.run(command, check=True, env=environment, cwd=ROOT)
        print(json.dumps({'built': str(target.relative_to(ROOT))}), flush=True)


if __name__ == '__main__':
    main()

