from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain
from conan.tools.files import (
    copy,
    download,
    apply_conandata_patches,
    export_conandata_patches,
    get,
)
from conan.tools.layout import basic_layout
from conan.tools.scm import Version, Git
import os

required_conan_version = ">=1.52.0"

# taken form conan-center-index recipe, until latest bug fix release is available, then delete this and move to main package


class CistaConan(ConanFile):
    name = "cista"
    description = (
        "Cista++ is a simple, open source (MIT license) C++17 "
        "compatible way of (de-)serializing C++ data structures."
    )
    license = "MIT"
    topics = ("cista", "serialization", "deserialization", "reflection")
    homepage = "https://github.com/felixguendling/cista"
    url = "https://github.com/conan-io/conan-center-index"
    package_type = "header-library"
    settings = "os", "arch", "compiler", "build_type"
    no_copy_source = True

    options = {"with_fmt": [True, False], "single-header": [True, False]}
    default_options = {"with_fmt": False, "single-header": False}

    @property
    def _min_cppstd(self):
        return 17

    @property
    def _compilers_minimum_version(self):
        return {
            "Visual Studio": "15.7" if Version(self.version) < "0.11" else "16",
            "msvc": "191" if Version(self.version) < "0.11" else "192",
            "gcc": "8",
            "clang": "6",
            "apple-clang": "9.1",
        }

    def requirements(self):
        if "with_fmt" in self.options and self.options.with_fmt:
            self.requires("fmt/[>=10.0.0]")

    def configure(self):
        if self.version in (f"0.{x}" for x in range(16)):
            self.options.rm_safe("with_fmt")

    def layout(self):
        cmake_layout(self)

    def validate(self):
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._min_cppstd)

        def loose_lt_semver(v1, v2):
            lv1 = [int(v) for v in v1.split(".")]
            lv2 = [int(v) for v in v2.split(".")]
            min_length = min(len(lv1), len(lv2))
            return lv1[:min_length] < lv2[:min_length]

        minimum_version = self._compilers_minimum_version.get(
            str(self.settings.compiler), None
        )
        if minimum_version and loose_lt_semver(
            str(self.settings.compiler.version), minimum_version
        ):
            raise ConanInvalidConfiguration(
                f"{self.name} {self.version} requires C++{self._min_cppstd}, which your compiler does not support.",
            )

    def source(self):
        for file in self.conan_data["sources"][self.version]:
            if "type" in file and file["type"] == "git":
                git = Git(self)
                git.clone(file["url"], target=".")
                git.checkout(file["commit"])
            else:
                if file["url"].endswith(".tar.gz"):
                    get(
                        self,
                        strip_root=True,
                        **file,
                    )
                else:
                    filename = os.path.basename(file["url"])
                    download(self, filename=filename, **file)
        apply_conandata_patches(self)

    def export_sources(self):
        export_conandata_patches(self)

    def build(self):
        cmake = CMake(self)
        args = {}
        args["CISTA_INSTALL"] = "ON"
        if "with_fmt" in self.options:
            args["CISTA_FMT"] = "ON" if self.options.with_fmt else "OFF"
        if self.settings.build_type == "Debug" and self.settings.compiler == "gcc":
            args["CISTA_ZERO_OUT"] = "ON"
        cmake.configure(variables=args)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.user_presets_path = False
        tc.generate()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libdirs = ["cista"]
