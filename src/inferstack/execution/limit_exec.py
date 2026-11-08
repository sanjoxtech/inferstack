"""Apply rlimits, then exec the candidate server: `python -m inferstack.execution.limit_exec CPU FILES FSIZE AS -- argv`."""
import os
import resource
import sys


def main(argv):
    cpu, files, fsize, address = (int(x) for x in argv[:4])
    if argv[4] != '--' or len(argv) < 6:
        raise SystemExit('usage: limit_exec CPU FILES FSIZE AS -- argv...')
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_NOFILE, (files, files))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    if address:
        resource.setrlimit(resource.RLIMIT_AS, (address, address))
    os.execvp(argv[5], argv[5:])


if __name__ == '__main__':
    main(sys.argv[1:])
