.. MPI Tutorials NGSolve Usermeeting 2019 documentation master file, created by
   sphinx-quickstart on Sun Jun 30 22:49:05 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MPI Tutorials NGSolve Usermeeting 2019
=================================================================

This mini-workshop is seperated into three parts:

Part one is a short introduction to MPI-parallel NGsolve in a jupyter notebook.

That notebook is not meant to be run by participants, as getting it to run with MPI
requires some additional setup. Instead a python-script containing the same code
is provided for you.

If you still want to run it yourself you need to install the "ipyparallel" package
and start up, and connect to, an ipython cluster.

Part two consists of a series of basic python scripts for you to experiment with.
This will hopefully get you used to NGsPy + MPI.

In part three, we will, step by step, build an MPI-parallel preconditioner for a magnetostatic
HCurl problem, using building blocks provided by NGSolve.

	  
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   mpi_basics.ipynb
   basic_examples.rst
   advanced_aux.ipynb

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
