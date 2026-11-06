from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain
from conan.tools.scm import Git
from conan.tools.files import chdir


class VallaRecipe(ConanFile):
    name = "valla"
    package_type = "library"
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

    def requirements(self):
        self.requires("abseil/20240116.2")
        self.requires("fmt/11.2.0")
        # Valla uses TBB headers and links to libtbb, but downstream consumers
        # should not be forced to link tbbmalloc/tbbmalloc_proxy. Export only the
        # headers transitively; final link closure is owned by the top-level
        # project.
        self.requires(
            "onetbb/2021.12.0", transitive_headers=True, transitive_libs=False
        )

    def validate(self):
        check_min_cppstd(self, "20")

    def source(self):
        git = Git(self)
        git.clone(
            "https://github.com/drexlerd/tree-compression.git", target="valla_src"
        )

        with chdir(self, "valla_src"):
            if len(str(self.version)) == 40:
                git.checkout(self.version)
            else:
                git.checkout(f"v{self.version}")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.user_presets_path = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure(build_script_folder="valla_src")
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["valla_core"]
