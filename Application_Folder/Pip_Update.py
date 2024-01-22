import subprocess
import sys


def update_dependencies():
    # Get list of outdated packages
    outdated_packages = subprocess.check_output ([sys.executable, '-m', 'pip', 'list', '--outdated']).decode ('utf-8')

    # Parse the output to get the package names
    outdated_packages = outdated_packages.split ('\n') [2:-1]
    outdated_packages = [pkg.split () [0] for pkg in outdated_packages]

    # Update each package
    for pkg in outdated_packages:
        subprocess.check_call ([sys.executable, '-m', 'pip', 'install', '--upgrade', pkg])


if __name__ == "__main__":
    update_dependencies ()
