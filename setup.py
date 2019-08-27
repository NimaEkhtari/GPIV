from setuptools import setup

setup(
    name='gpiv',
    version='0.1',
    description='Geospatial PIV with uncertainty propagation',
    author='Preston Hartzell',
    author_email='preston.hartzell@gmail.com',
    py_modules=['gpiv', 'piv_functions', 'show_functions'],
    entry_points='''
        [console_scripts]
        gpiv=gpiv:cli
    ''',
)
