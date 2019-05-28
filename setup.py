from setuptools import setup

setup(name='gpiv',
      version='0.0',
      description='PIV tailored to the geospatial community',
      url='https://bitbucket.org/pjh172/gpiv/src/master/',
      author='Preston Hartzell',
      author_email='preston.hartzell@gmail.com',
      packages=['gpiv'],
      install_requires=[
          'docopt',
          'rasterio',
          'numpy',
          'shapely',
          'scikit-image',
          'matplotlib'
      ],
      entry_points={
          'console_scripts':[
              'gpiv=gpiv.__main__:main'
          ]
      })