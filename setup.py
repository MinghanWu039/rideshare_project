from setuptools import setup, find_packages

setup(
    name='z3_rideshare_planner',
    version='0.1.0',
    author='Travis Wu',
    author_email='traviswu0524@gmail.com',
    description='A rideshare planner that gives an optimal plan for multiple drivers and passengers using Z3',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MinghanWu039/z3_rideshare_planner',
    packages=find_packages(),
    install_requires=[
        [
            'googlemaps',
            'pandas',
            'matplotlib',
            'datetime',
            'folium',
            'z3-solver',
            'polyline',
            'numpy',
        ]
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)