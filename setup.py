try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'gpiv',
    'description': 'PIV tailored to the geospatial community with robust error estimation.',
    'author': 'Preston Hartzell',
    'author_email': 'preston.hartzell@gmail.com',
    'version': '0.1',
    'install_requires': [
        'nose',
        'Click',
    ],
    'py_modules': ['gpiv'],
    entry_points='''
        [console_scripts]
        gpiv=gpiv:cli
    ''',
}

setup(**config)