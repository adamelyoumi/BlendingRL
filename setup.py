from setuptools import setup, find_packages

setup(
    name='bla',
    version='0.1',
    packages=find_packages("C:/Users/adame/OneDrive/Bureau/CODE/BlendingRL/blendv"),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'your_command=your_module:main_function',
        ],
    },
)