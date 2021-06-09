from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='deepl',
    url='https://github.com/akanametov/deepl',
    author='Azamat Kanametov',
    author_email='akkanametov@gmail.com',
    # Needed to actually package something
    packages=['deepl'],
    # Needed for dependencies
    #install_requires=['numpy==1.18.0', 'scikit-learn==0.21.2'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
