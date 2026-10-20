"""Allow `python -m gs_merger` to run the workflow."""

from gs_merger.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
