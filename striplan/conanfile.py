from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


class BeIwRgnnRecipe(ConanFile):
    name = "striplan"
    version = "0.0.1"
    package_type = "library"
    settings = "os", "compiler", "build_type", "arch"

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

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_benchmarks": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_benchmarks": False,
        "cista/*:with_fmt": True,
        "hwloc/*:shared": True,
    }
    default_options.update({f"boost/*:without_{comp}": True for comp in BOOST_COMPS})
    for comp in ("iostreams", "random", "regex", "system"):
        default_options.update({f"boost/*:without_{comp}": False})

    def export_sources(self):
        self.copy("conandata.yml")

    def requirements(self):
        self.requires("boost/1.86.0")
        self.requires("zlib/1.3.1", override=True)
        self.requires("abseil/20240116.2", override=True)
        self.requires("fmt/11.2.0")
        self.requires("range-v3/0.12.0")
        self.requires("nanobind/2.9.2")
        self.requires("strong_type/v10")
        if self.options.with_benchmarks:
            self.requires("argparse/3.2")
            self.requires("benchmark/1.8.3")

        if self.conan_data and "requirements" in self.conan_data:
            for requirement in self.conan_data["requirements"]:
                self.requires(requirement)
        else:
            self.requires("dlpack/1.2")
            self.requires("loki/02880ea1d4ce4ca11deca188050ea6cc83f88dbb")
            self.requires("nauty/2.8.8")
            self.requires("gtest/1.14.0")
            self.requires("unordered_dense/4.8.1")
            self.requires("parallel-hashmap/1.3.11")
            self.requires("valla/f487d3fcba00b88bba1e88d9e111747f53679da1")

    def validate(self):
        check_min_cppstd(self, "20")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.set_property("nanobind", "cmake_find_mode", "both")
        deps.generate()

        toolchain = CMakeToolchain(self)
        toolchain.user_presets_path = False
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
