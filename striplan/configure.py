#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_mimir_prefix() -> str | None:
    try:
        import pymimir

        if hasattr(pymimir, "get_cmake_dir"):
            return str(pymimir.get_cmake_dir())
        mimir_path = Path(pymimir.__file__).resolve().parent
        cmake_path = mimir_path / "lib" / "cmake" / "mimir"
        if cmake_path.exists():
            return str(cmake_path)
    except ImportError:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure striplan")
    parser.add_argument("--noconan", action="store_true", help="Disable Conan")
    parser.add_argument(
        "--only_install", action="store_true", help="Only install dependencies"
    )
    parser.add_argument("--deps_policy", default="missing", help="Conan build policy")
    parser.add_argument("--cmake_cmd", default="cmake", help="CMake executable")
    parser.add_argument("--conan_cmd", default="conan", help="Conan executable")
    parser.add_argument("--build_dir", default="build", help="Build directory")
    parser.add_argument("--source_dir", default=".", help="Source directory")
    parser.add_argument(
        "--config", default="Release", help="Build type (Debug, Release, etc.)"
    )
    parser.add_argument("--toolchain_file", help="CMake toolchain file")
    parser.add_argument(
        "--with_benchmarks",
        action="store_true",
        help="Enable benchmark dependencies and CMake targets",
    )

    args, extra_args = parser.parse_known_args()

    use_conan = not args.noconan
    build_dir = Path(args.build_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    script_dir = Path(__file__).resolve().parent

    print(f"Configuring striplan: config={args.config}, conan={use_conan}")

    cmake_toolchain_file = args.toolchain_file
    if use_conan:
        conan_home = Path(
            os.environ.get("CONAN_HOME", Path.home() / ".conan2")
        ).resolve()
        default_profile = conan_home / "profiles" / "default"
        if not default_profile.exists():
            subprocess.run([args.conan_cmd, "profile", "detect", "--force"], check=True)

        conan_export_script = script_dir / "conan_export.py"
        if conan_export_script.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(conan_export_script),
                    f"--conan_cmd={args.conan_cmd}",
                ],
                check=True,
            )

        conan_install_dir = build_dir / "conan"
        conan_args = [
            "-s",
            f"build_type={args.config}",
            "-s:h",
            "compiler.cppstd=gnu23",
            "-s:b",
            "compiler.cppstd=gnu23",
            "--profile:host=default",
            "--profile:build=default",
            "--options=loki/*:fPIC=True",
            "--options=nauty/*:fPIC=True",
            f"--build={args.deps_policy}",
        ]
        if args.with_benchmarks:
            conan_args.append("--options=striplan/*:with_benchmarks=True")

        install_cmd = [
            args.conan_cmd,
            "install",
            str(source_dir),
            f"-of={conan_install_dir}",
            "-s",
            f"&:build_type={args.config}",
        ] + conan_args

        print(f"Running Conan: {' '.join(install_cmd)}")
        subprocess.run(install_cmd, check=True)

        if not cmake_toolchain_file:
            possible_path = (
                conan_install_dir
                / "build"
                / args.config
                / "generators"
                / "conan_toolchain.cmake"
            )
            possible_path_flat = conan_install_dir / "conan_toolchain.cmake"
            if possible_path.exists():
                cmake_toolchain_file = str(possible_path)
            elif possible_path_flat.exists():
                cmake_toolchain_file = str(possible_path_flat)

    if args.only_install:
        return

    cmake_args = [
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DCMAKE_BUILD_TYPE={args.config}",
    ]
    if args.with_benchmarks:
        cmake_args.append("-DSTRIPLAN_BUILD_BENCHMARKS=ON")
    if cmake_toolchain_file:
        cmake_args.append(f"-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}")

    mimir_prefix = find_mimir_prefix()
    if mimir_prefix:
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={mimir_prefix}")
        print(f"Found mimir CMake package at: {mimir_prefix}")
    else:
        print("Warning: pymimir.get_cmake_dir() was not available.")

    print(
        f"Running CMake: {args.cmake_cmd} {' '.join(cmake_args)} {' '.join(extra_args)}"
    )
    subprocess.run([args.cmake_cmd] + cmake_args + extra_args, check=True)


if __name__ == "__main__":
    main()
