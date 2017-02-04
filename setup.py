from distutils.core import setup

setup(
    name='mechpy',
    version='0.1',
    description="mechanical engineering toolbox",
    author='Neal Gordon',
    author_email='nealagordon@gmail.com',
    packages=['mechpy'],
	package_data={'': ['compositematerials.csv']},
	include_package_data=True,
    license="The MIT License (MIT)",
    long_description=open('README.md').read(),
    url='https://github.com/nagordon/mechpy',
    keywords = ['composites', 'mechanics', 'statics', 'materials'],
    classifiers = [
      "Programming Language :: Python",
      "Programming Language :: Python :: 3.4",
      "License :: The MIT License (MIT)",
      "Operating System :: OS Independent",
      "Intended Audience :: Science/Research",
      "Topic :: Scientific/Engineering",
      "Development Status :: 3 - Alpha"
    ],
    install_requires=['numpy', 'matplotlib', 'scipy','sympy','pint','python-quantities'],
)
