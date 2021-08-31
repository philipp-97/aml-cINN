from os.path import join
from waflib.extras.test_base import summary
from waflib.extras.symwaf2ic import get_toplevel_path


def depends(ctx):
    ctx("freia")


def options(opt):
    pass


def configure(cfg):
    cfg.load("python")
    cfg.check_python_version()


def build(bld):
    bld(name="aml-cinn",
        source=bld.path.ant_glob("src/py/**/*.py"),
        features="use py",
        install_path="${PREFIX}/lib",
        relative_trick=True,
        install_from="src/py")

    bld.add_post_fun(summary)
