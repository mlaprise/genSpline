#!/usr/bin/env python

METADATA = dict(
	name='genSpline',
	version='0.1',
	author='Martin Laprise',
	author_email='martin.laprise.1@ulaval.ca',
	description='Spline-based genetic optimization class ',
	long_description=open('README.markdown').read(),
	url='http://github.com/mlaprise/genSpline',
	license = 'MIT License',
	keywords='python genetic spline optimization',
)
SETUPTOOLS_METADATA = dict(
	install_requires=['numpy', 'scipy'],
	extras_require = dict(plot = ['matplotlib >= 0.99.0']),
	include_package_data=True,
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'Intended Audience :: Developers',
		'Intended Audience :: Education',
		'License :: OSI Approved :: MIT License',
		'Topic :: Software Development :: Libraries :: Python Modules',
		'Topic :: Scientific/Engineering',
		'Operating System :: OS Independent',
		'Programming Language :: Python',
	],
	packages=['python-genspline'],
)

if __name__ == '__main__':
	try:
		import setuptools
		METADATA.update(SETUPTOOLS_METADATA)
		setuptools.setup(**METADATA)
	except ImportError:
		import distutils.core
		distutils.core.setup(**METADATA)
