from setuptools import setup, find_packages 
from typing import List


HYPEN_E_DOT = '-e .'

def get_requirements(file_path=str)->List[str]:

    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [r.replace("\n","") for r in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements        




setup(
    name="charges_prediction",
    version="0.0.1",
    description="This project aims to predict insurance charges",
    author="Pritam",
    author_email="pritamnarwade11@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
    