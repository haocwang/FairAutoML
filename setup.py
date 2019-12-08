from distutils.core import setup
setup(
  name = 'FairAutoML',         # How you named your package folder
  packages = ['FairAutoML'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A Fairness-aware AutoML System',   # Give a short description about your library
  author = 'Haochen Wang',                   # Type in your name
  author_email = 'hw1985@nyu.edu',      # Type in your E-Mail
  url = 'https://github.com/user/haocwang',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/haocwang/FairAutoML/archive/v0.1-alpha.tar.gz',    # I explain this later on
  keywords = ['autoML', 'fairness', 'Machine Learning'],   # Keywords that define your package best
  install_requires=[            
          'numpy',
          'pandas',
          'matplotlib',
          'auto-sklearn',
          'aif360',
          'tqdm',
          'pyrfr',
          'cvxpy',
          'numba',
          'BlackBoxAuditing',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)