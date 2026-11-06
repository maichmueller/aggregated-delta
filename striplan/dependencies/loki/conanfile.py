from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain
from conan.tools.scm import Git

# Boost components from its conanfile recipe
# https://github.com/conan-io/conan-center-index/blob/master/recipes/boost/all/conanfile.py
BOOST_COMPS = (
    "atomic",
    "charconv",
    "chrono",
    "cobalt",
    "container",
    "context",
    "contract",
    "coroutine",
    "date_time",
    "exception",
    "fiber",
    "filesystem",
    "graph",
    "graph_parallel",
    "iostreams",
    "json",
    "locale",
    "log",
    "math",
    "mpi",
    "nowide",
    "process",
    "program_options",
    "python",
    "random",
    "regex",
    "serialization",
    "stacktrace",
    "system",
    "test",
    "thread",
    "timer",
    "type_erasure",
    "url",
    "wave",
)


class LokiRecipe(ConanFile):
    name = "loki"
    package_type = "static-library"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }

    default_options = {
        "shared": False,
        "fPIC": True,
    }

    default_options.update({f"boost/*:without_{comp}": True for comp in BOOST_COMPS})
    default_options.update(
        {f"boost/*:without_iostreams": False for comp in BOOST_COMPS}
    )
    for comp in ("iostreams", "random", "regex", "system"):
        default_options.update({f"boost/*:without_{comp}": False})

    exports_sources = "fix_includes.patch", "disable_exe.patch"

    def requirements(self):
        self.requires("boost/[>=1.86.0]")
        self.requires("abseil/20240116.2")
        self.requires("fmt/11.2.0")

    def source(self):
        git = Git(self)
        git.clone("https://github.com/drexlerd/Loki.git", target="loki_src")

        # Use conan.tools.files.chdir instead of self.chdir
        from conan.tools.files import chdir, patch

        with chdir(self, "loki_src"):
            if len(str(self.version)) == 40:
                # Checkout commit hash
                git.checkout(self.version)
            else:
                git.checkout(f"v{self.version}")

        # Apply patches
        patch(self, base_path="loki_src", patch_file="fix_includes.patch")
        patch(self, base_path="loki_src", patch_file="disable_exe.patch")

    def configure(self):
        # if windows remove the fPIC option, it has no effect
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def validate(self):
        # loki requires C++20
        check_min_cppstd(self, "20")

    def build(self):
        cmake = CMake(self)
        cmake.configure(build_script_folder="loki_src")
        cmake.build()

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.user_presets_path = False
        tc.generate()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["loki_parsers"]
