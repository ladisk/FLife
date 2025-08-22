import os
import sys
import tomli
import tomli_w
import argparse

package_name = "FLife"

def synchronize_version():
    print("Synchronizing version (pyproject.toml and __init__.py)...")

    # Read the version from pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    
    version_toml = pyproject["project"]["version"]

    # Read the __init__.py
    with open(f"{package_name}/__init__.py", "r") as f:
        init = f.readlines()

    # Replace the version with the one from pyproject.toml
    for i, line in enumerate(init):
        if "__version__" in line:
            init[i] = "__version__ = " + f'"{version_toml}"' + "\n"
    init = "".join(init)

    # Write the new __init__.py
    with open(f"{package_name}/__init__.py", "w") as f:
        f.write(init)

    # Update docs/source/conf.py
    with open("docs/source/conf.py", "r", encoding="utf8") as f:
        conf = f.readlines()

    for i, line in enumerate(conf):
        if "version = " in line and not line.strip().startswith("#"):
            conf[i] = f"version = '{version_toml.rsplit('.', 1)[0]}'\n"
        elif "release = " in line and not line.strip().startswith("#"):
            conf[i] = f"release = '{version_toml}'\n"

    # Write the new conf.py
    with open("docs/source/conf.py", "w", encoding="utf8") as f:
        f.write("".join(conf))

def set_version(version):
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    pyproject["project"]["version"] = version
    with open("pyproject.toml", "wb") as f:
        tomli_w.dump(pyproject, f)

def bump_version(bump):
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    version = pyproject["project"]["version"]
    version_parts = version.split(".")
    if bump == "patch":
        version_parts[2] = str(int(version_parts[2]) + 1)
    elif bump == "minor":
        version_parts[1] = str(int(version_parts[1]) + 1)
        version_parts[2] = "0"
    elif bump == "major":
        version_parts[0] = str(int(version_parts[0]) + 1)
        version_parts[1] = "0"
        version_parts[2] = "0"
    else:
        raise ValueError(f"Invalid bump type: {bump}")
    
    version = ".".join(version_parts)
    set_version(version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bump", default="", choices=["patch", "minor", "major"], help="Bump the version of the package.")
    parser.add_argument("--set-version", type=str, help="Set the version of the package.")
    args = parser.parse_args()

    if args.set_version:
        set_version(args.set_version)

    elif args.bump:
        bump_version(args.bump)

    synchronize_version()
