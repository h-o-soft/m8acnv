from setuptools import setup, find_packages

setup(
	name="m8acnv",
	version="0.1.0",
	install_requires=["Pillow"],
	packages=find_packages(),
	entry_points={
		"console_scripts": [
			"m8acnv=m8acnv.app:main"
		]
	}
)
