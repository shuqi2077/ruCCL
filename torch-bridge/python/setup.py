from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildCollectives(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parents[3]/'ruda-torch/python/ruda_torch'
        if not (source/'distributed_training.py').is_file():
            raise RuntimeError('build ruccl-bridge from the RUDA source checkout')
        destination = Path(self.build_lib)/'ruda_collectives'
        for module in source.glob('*.py'):
            if module.name != '__init__.py':
                shutil.copy2(module, destination/module.name)


setup(cmdclass={'build_py': BuildCollectives})
