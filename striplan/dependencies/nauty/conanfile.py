from conan import ConanFile
from conan.tools.files import get
from conan.tools.gnu import AutotoolsToolchain, Autotools, PkgConfig


class NautyRecipe(ConanFile):
    name = "nauty"
    version = "2.8.8"
    package_type = "static-library"
    settings = "os", "compiler", "build_type", "arch"
    generators = ["CMakeDeps"]

    options = {
        "fPIC": [True, False],
    }

    default_options = {
        "fPIC": True,
    }

    def source(self):
        get(
            self,
            "https://users.cecs.anu.edu.au/~bdm/nauty/nauty2_8_8.tar.gz",
            strip_root=True,
        )

    def configure(self):
        # if windows remove the fPIC option, it has no effect
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def build(self):
        ac = Autotools(self)
        ac.configure()
        ac.make()

    def generate(self):
        tc = AutotoolsToolchain(self)
        tc.configure_args.append("--enable-tls")
        tc.configure_args.append("--enable-generic")
        tc.generate()

    def package(self):
        ac = Autotools(self)
        ac.install()

    def package_info(self):
        self.cpp_info.libs = ["nauty"]
