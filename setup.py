from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
import logging


def post_install():
    try:
        import nltk

        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
    except ModuleNotFoundError as e:
        logging.exception(e)


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        post_install()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        post_install()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        post_install()


with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="actableai-lib",
    version="0.1.1",
    author="Actable AI Team",
    author_email="trung@actable.ai",
    description="Actable AI Machine Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Actable-AI/actableai-lib",
    project_urls={
        "Doc": "https://lib.actable.ai",
        "App": "https://app.actable.ai",
        "Website": "https://actable.ai",
    },
    packages=find_packages(where="."),
    python_requires=">=3.8, <3.9",
    install_requires=[
        "scikit-learn>=1.0.2",
        "pandasql>=0.7.3",
        "ray[tune,default,serve]>=1.13.0",
        "autogluon.tabular[all]>=0.5.0",
        "autogluon.text>=0.5.0",
        "gluonts[Prophet,R]>=0.10.3",
        "econml>=0.13.0",
        "ipython>=7.32.0",
        "hyperopt>=0.2.7",
        "ujson>=4.0.2",
        "python-rapidjson>=1.4",
        "visions>=0.7.1",
        "shap>=0.39.0",
        "nltk>=3.7",
        "pyabsa>=1.14.7",
        "flair>=0.11.3",
        "mlxtend>=0.20.0",
        "aioredis>=1.3.1",
        "redis>=3.5.3",
        "river>=0.8.0",
    ],
    extras_require={
        "cpu": [
            "mxnet>=1.8.0.post0",
            "tensorflow>=2.9.1",
            "torch==1.10.1+cpu",
            "torchvision==0.11.2+cpu",
            "torchaudio>=0.10.1",
        ],
        "gpu": [
            "mxnet-cu110>=1.8.0.post0",
            "tensorflow-gpu>=2.9.1",
            "torch==1.10.1+cu111",
            "torchvision==0.11.2+cu111",
            "torchaudio>=0.10.1",
        ],
        "dev": [
            "flake8==3.9.2",
            "pytest==7.1.1",
            "pytest-xdist==2.5.0",
            "black==22.8.0",
        ],
    },
    include_package_data=True,
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
    # https://pypi.org/classifiers/
    classifiers=[
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "Environment :: MacOS X",
        "Topic :: System :: Operating System Kernels :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix",
    ],
)
