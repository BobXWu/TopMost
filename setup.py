from setuptools import setup, find_packages


with open("./README.md", "r") as file:
    LONG_DESCRIPTION = file.read()


VERSION = '0.0.1'
DESCRIPTION = 'Topmost: Neural Topic Modeling System Tookit'


# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="topmost",
        version=VERSION,
        author="Xiaobao Wu",
        author_email="xiaobao002@e.ntu.edu.sg",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url='https://github.com/bobxwu/',
        packages=find_packages(),
        # packages=find_packages(include=['topmost', 'topmost.*']),
        license="Apache 2.0 License",
        install_requires=[], # add any additional packages that needs to be installed along with your package. Eg: 'caer'
        keywords=['toolkit', 'topic model', 'neural topic model'],
        include_package_data=True,
        package_data={
            'topmost': ['../datasets/stopwords/*.txt']
        },
        classifiers= [
            'Development Status :: 3 - Alpha',
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
