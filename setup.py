from setuptools import setup, find_packages


with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('README.rst') as f:
    readme = f.read()

VERSION = '0.0.5'
DESCRIPTION = 'Topmost: A Topic Modeling System Tookit'
LONG_DESCRIPTION = """
    [TopMost](https://github.com/bobxwu/topmost) provides complete lifecycles of topic modeling, including datasets, models, training, and evaluations.
    It covers the most popular topic modeling scenarios, like basic, dynamic, hierarchical, and cross-lingual topic modeling.
"""


# Setting up
setup(
        name="topmost",
        version=VERSION,
        author="Xiaobao Wu",
        author_email="xiaobao002@e.ntu.edu.sg",
        description=DESCRIPTION,
        long_description=readme,
        long_description_content_type="text/x-rst",
        # long_description_content_type="text/markdown",
        url='https://github.com/bobxwu/topmost',
        packages=find_packages(),
        # packages=find_packages(include=['topmost', 'topmost.*']),
        license="Apache 2.0 License",
        install_requires=requirements,
        keywords=['toolkit', 'topic model', 'neural topic model'],
        include_package_data=True,
        test_suite='tests',
        classifiers= [
            'Development Status :: 3 - Alpha',
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
