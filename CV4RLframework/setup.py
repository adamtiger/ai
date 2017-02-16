from setuptools import setup, find_packages

setup(name='cv4RLframework',
      version='0.1',
      description='Framework for training the AI algorithms in cv4sensorhub',
      url='http://adamtiger.github.io/ai',
      author='Adam Budai',
      author_email='budai8adam@gmail.com',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('cv4rl')],
      zip_safe=False)