from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        entry_points={"console_scripts": ["scl=cli.cli:scl"]},
        name="",
        version="0.1",
        packages=find_packages()
    )
