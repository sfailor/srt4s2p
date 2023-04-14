from setuptools import setup, find_packages

setup(
    name='srt4s2p',
    version='0.0.1',
    author='Samuel W. Failor',
    author_email='samuel.failor@gmail.com',
    url='http://github.com/sfailor/srt4s2p',
    install_requires=   [
                            'pyqtgraph',
                            'PyQt5',
                            'key-point-finder',
                            'numpy',
                            'opencv-python',
                            'pandas',
                            'scipy',
                            'matplotlib'                           
                        ],
    packages=find_packages(),
    classifiers=[
                    "Programming Language :: Python :: 3",
                    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                     "Operating System :: OS Independent",
                ],
)

