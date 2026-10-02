# -*- coding: utf-8 -*-
import os
import sys


def main():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gantt_chart.py")
    argv = ["python3", script] + sys.argv[1:]
    try:
        os.execvp("python3", argv)
    except OSError:
        sys.stderr.write("python3 not found\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
