import setuptools

setuptools.setup(
    name='protein_holography_pytorch',
    version='0.1.0',
    author='Michael Pun',
    author_email='gvisan01@.cs.washington.edu',
    description='learning protein neighborhoods by incorporating rotational symmetry - pytorch version',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/StatPhysBio/protein_holography-pytorch',
    python_requires='>=3.9',
    install_requires='',
    packages=setuptools.find_packages(),
)
