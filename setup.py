from distutils.core import setup

setup(
    name='mechpy',
    version='0.01',
    description="A Python package for mechanical engineers",
    author='Neal Gordon',
    author_email='nealagordon@gmail.com',
    packages=['mechpy'],
    license="The MIT License (MIT)",
    long_description=open('README.md').read(),
    url='https://github.com/nagordon/mechpy',
    keywords = ['composites', 'mechanics', 'statics', 'materials'],
    classifiers = [
      "Programming Language :: Python",
      "Programming Language :: Python :: 3.4",
      "License :: OSI Approved :: GNU General Public License (GPL)",
      "Operating System :: OS Independent",
      "Intended Audience :: Science/Research",
      "Topic :: Scientific/Engineering",
      "Development Status :: 2 - Pre-Alpha"
    ],
    install_requires=['numpy', 'matplotlib', 'scipy','sympy','pint','python-quantities'],
)