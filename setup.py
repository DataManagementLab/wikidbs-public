import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="wikidbs",
    
    description="Crawling a dataset of relational databases from wikidata",

    author="Liane Vogel, Jan-Micha Bodensohn",

    packages=['wikidbs'],

    long_description=read('README.md'),
)