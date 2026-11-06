#!/usr/bin/python3

import argparse
import os
import subprocess
import sys

# Parse arguments
parser = argparse.ArgumentParser(description="Export dependencies using Conan.")
parser.add_argument(
    "--conan_cmd",
    type=str,
    default="conan",
    help="Command to invoke Conan (default: 'conan').",
)
args = parser.parse_args()

# Conan command
conan_cmd = args.conan_cmd


# Read versions from conandata.yml
def read_versions_from_conandata(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"conandata.yml not found at {file_path}")

    dependencies = {}
    with open(file_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        try:
            req_start = lines.index("requirements:")
        except ValueError:
            raise ValueError("No 'requirements:' section found in conandata.yml.")
        for line in lines[req_start + 1 :]:
            line = line.strip()
            if not line:
                break
            if line.startswith("-"):
                try:
                    dep, version = str(line[1:].strip().strip('"')).split("/")
                    dependencies[dep.strip()] = version.strip()
                except ValueError:
                    print(f"Warning: Skipping malformed line: {line}")
    return dependencies


conandata_path = "conandata.yml"


try:
    dependency_versions = read_versions_from_conandata(conandata_path)
except Exception as e:
    print(f"Error reading conandata.yml: {e}")
    exit(1)


dependencies = """
loki
nauty
cista
valla
"""


# export dependencies
had_errors = False
for dep in (dep.strip() for dep in dependencies.splitlines() if dep):
    version = dependency_versions.get(dep)
    if not version:
        print(f"Warning: Version for dependency '{dep}' not found in conandata.yml.")
        continue

    try:
        subprocess.run(
            [conan_cmd, "export", f"dependencies/{dep}", f"--version={version}"],
            check=True,
        )
        print(f"Successfully exported {dep} version {version}.")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting {dep} version {version}: {e}")
        had_errors = True

if had_errors:
    sys.exit(1)
