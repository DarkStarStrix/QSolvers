import subprocess
import sys


def update_dependencies():
    outdated_packages = subprocess.check_output ([sys.executable, '-m', 'pip', 'list', '--outdated']).decode (
        'utf-8').split ('\n') [2:-1]
    for pkg in [pkg.split () [0] for pkg in outdated_packages]:
        subprocess.check_call ([sys.executable, '-m', 'pip', 'install', '--upgrade', pkg])


if __name__ == "__main__":
    update_dependencies ()
