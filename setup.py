from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the requirements by reading the content of requirements.txt
    and removing newline characters.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


def process_requirements(requirements: List[str]) -> List[str]:
    """
    This function processes the requirements list and removes the '-e .' entry if present.
    """
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Solomon',
    author_email='sholasolomon@yahoo.com',
    packages=find_packages(),
    install_requires=process_requirements(get_requirements('requirements.txt')),
    include_package_data=True,
    zip_safe=False
)



# from setuptools import find_packages, setup
# from typing import List

# HYPEN_E_DOT = '-e .'

# def get_requirements(file_path:str)->List[str]:
#     '''
#     This function will return requirements

#     the code below reads the content of requirements.txt, it replace \ with empty space
#     '''
#     requirements=[]
#     with open(file_path) as file_obj: 
#         requirements=file_obj.readlines()
#         [req.replace("\n", "")for req in requirements]


#         if HYPEN_E_DOT in requirements:
#             requirements.remove(HYPEN_E_DOT)

#     return requirements



# setup(
# name='mlproject',
# version='0.0.1',
# author='Solomon',
# author_email='sholasolomon@yahoo.com',
# packages=find_packages(),
# install_requires=get_requirements('requirements.txt')
# )