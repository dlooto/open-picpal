import datetime
from setuptools import setup, find_packages


setup(
    name='open-picpal',
    version='0.1.0',
    author='Jon. Shon',
    author_email='xjbean@email.com',
    description='An open-source tool for image training and automatic classification',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.11.0',
        'keras==2.11.0'
    ],
    options={'build': {'force': True}},
    long_description='Released on ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.',
    long_description_content_type='text/plain',
)

