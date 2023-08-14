Introduction
------------

Added Mass allows to calculate the added masses for automatically generated 
shapes using the Salome Meca software (in particular the Code_Aster solver).

Shapes are generated based on Viquerat's github work (https://github.com/jviquerat/shapes).



Usage
-----

Step 1 : Generate the shapes dataset (does not require Salome Meca)

python generate_dataset.py


Step 2 : Calculate the added masses using Code_Aster (requires Salome Meca)

python calculate.py

Step 3 : Train the model to fit to the calculated reference data

python train.py
