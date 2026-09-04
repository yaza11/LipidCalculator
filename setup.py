from setuptools import setup, find_packages

setup(
    name='LipidCalculator',
    version='0.2.13',
    url='https://github.com/yaza11/LipidCalculator',
    author='Yannick Zander',
    author_email='yannick.zander@gmail.com',
    description='Toolbox for dealing with Intact polar lipids (IPLs)',
    install_requires=['numpy', 'matplotlib', 'pandas', 'sympy', 'rdkit',
                      'requests', 'tqdm', 'IsoSpecPy', 'scipy',
                      # 'msIO'  # take out requirement for now
                      ],
    packages=find_packages(),
    include_package_data=True,
)

# pip install git+https://github.com/yaza11/msIO.git
