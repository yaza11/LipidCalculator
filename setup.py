from setuptools import setup, find_packages

setup(
    name='LipidCalculator',
    version='0.0.1',
    url='',
    author='Yannick Zander',
    author_email='author@gmail.com',
    description='Toolbox for dealing with Intact polar lipids (IPLs)',
    install_requires=['numpy', 'matplotlib', 'pandas', 'sympy', 'rdkit',
                      'requests', 'tqdm', 'IsoSpecPy', 'scipy', 'msIO'],
    packages=find_packages(),
    include_package_data=True,
)
