from setuptools import setup, find_packages

setup(  name = 'udiff',
        
        version = '0.0.1',
        
        description = 'Module for smoothing and differentiating time series batch data.',
        
        author = 'Tim Forster',
        
        packages = find_packages(),
        
        package_data = {'': ['priors/*.dat']},
        
        install_requires = [    'numpy',
                                'scipy',
                                'matplotlib',
                                'seaborn',
                                'scikit-learn',
                                'prettytable'
                            ],
        
        zip_safe = False
    
    )