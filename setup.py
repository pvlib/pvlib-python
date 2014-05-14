from distutils.core import setup
 
setup(
    name='pvlpy',
    version='1.0',
	author='Clifford Hanson and Rob Andrews',
	author_email='Rob.Andrews@calamaconsulting.ca',
	packages=['pvlpy','pvlpy.test'],
    license='The BSD 3-Clause License',
    long_description=open('README.txt').read(),
)