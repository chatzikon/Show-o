import argparse
import importlib
import sys


def run_fusion_version(v1_main, v2_module, stage=None):
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--fusion-version",
        type=int,
        choices=[1, 2],
        default=2,
    )

    options, remaining = parser.parse_known_args()

    # Remove only our version flag before the selected script
    # parses its own arguments.
    original_argv = sys.argv[:]

    try:
        sys.argv = [original_argv[0], *remaining]

        if options.fusion_version == 1:
            return v1_main()

        if stage is not None:
            # The existing training entry point determines
            # whether this is Stage 1 or Stage 2.
            sys.argv.extend(["--stage", str(stage)])

        module = importlib.import_module(v2_module)
        return module.main()

    finally:
        sys.argv = original_argv