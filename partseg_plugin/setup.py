from setuptools import setup

setup(name='PartSeg-neutrofile',
      version='0.1',
      description='Neutrofile analysis plugin',
      url=' git@github.com:Czaki/Neutrofile_analysis.git',
      author="Grzegorz Bokota",
      author_email="g.bokota@cent.uw.edu.pl",
      license='MIT',
      packages=['neutrofile_plugin'],
      zip_safe=False,
      entry_points={
            'PartSeg.plugins': [".neutrofile_analysis = neutrofile_plugin"]
      },
      install_requires=["PartSeg"])
