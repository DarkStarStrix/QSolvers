import subprocess
import argparse
import plotly.graph_objects as go


def get_pip_info(package_name, command):
    cmd = f"{command} {package_name}"
    output = subprocess.check_output (cmd.split ()).decode ("utf-8")
    return [x.split ("==") if command == 'pipdeptree' else x.split (": ") for x in output.split ("\n") if x]


def create_figure(pip_info):
    fig = go.Figure (data=go.Scatter (x=[x [0] for x in pip_info], y=[x [1] for x in pip_info], mode="markers"))
    return fig


def main():
    parser = argparse.ArgumentParser (description="Visualize pip info of a package")
    parser.add_argument ("package_name", type=str, help="The package name to visualize")
    args = parser.parse_args ()

    pip_tree = get_pip_info (args.package_name, 'pipdeptree')
    pip_dependencies = get_pip_info (args.package_name, 'pip show')

    pip_info = pip_tree + pip_dependencies
    fig = create_figure (pip_info)
    fig.show ()


if __name__ == "__main__":
    main ()
