Getting started
===============

This program is intended to be run two ways. 

1. The first offers the most functionality and as such is preferable. 

2. The second allows the program to be run using the "make" command which could be a faster solution. 

1. By running "pip install -e ." in the root folder "/woven_planet", the program will be installed as a package. 
        This package can be accessed in python or ipython by importing it as follows: "from src.predict_model import run_model".
        The model can then be run in a python interactive environment by calling the "run_model()" method. 
        If the models runs successfully, this should return "Model accuracy score is 0.68~"
    Once the package has been installed, any of the other methods and functions from the other python files can be accessed and used as needed. 
    The "predict(model, observation)" method in the predict_model module can be used to predict whether a student will pass or fail.

2. By running "make" in the root directory "/woven_planet" the Makefile will compile and run the program, outputting the accuracy score of the model.
    This is dependent on the necessary requirements being installed. These can be found in the "requirements.txt"
    If the models runs successfully, this should return "Model accuracy score is 0.68~"


