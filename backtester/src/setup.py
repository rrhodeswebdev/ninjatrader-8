from distutils.core import setup

with open('README.md', 'r', encoding="utf-8") as readmefile:
    readme = readmefile.read()


setup(
    name = 'backintime',
    version = '20241112',
    description = 'Tool for testing trading strategies on historical data',
    author='Akim Mukhtarov',
    author_email = 'akim.int80h@gmail.com',
    packages = [
        'backintime',
        'backintime/analyser',
        'backintime/analyser/indicators',
        'backintime/broker',
        'backintime/broker/default',
        'backintime/broker/futures',
        'backintime/data',
        'backintime/result',
        'backintime/declarative',
        'backintime/declarative/indicators'
        ],
    install_requires = [
        'numpy>=1.26.0,<2.0.0',
        'pandas>=2.2.0',
        'python-dateutil>=2.8.2',
        'pytz>=2021.3',
        'requests>=2.27.1',
        'urllib3>=2.0.0'
        ],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    long_description_content_type='text/markdown',
    long_description=readme,
)
