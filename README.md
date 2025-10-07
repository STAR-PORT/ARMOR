# Installing and running ARMOR

## 0 - Requirements

This software requires python3 and the following libraries:
  - math: https://docs.python.org/3/library/math.html
  - numpy: https://numpy.org/
  - matplotlib: https://matplotlib.org/stable/index.html
  - scipy: https://scipy.org/install/
  - tkinter: https://docs.python.org/3/library/tkinter.html
  - (optional) jupyterlab: https://jupyter.org/install

## 1 - From source code (Python 3)

### Download source code with Git
The source code is available from
[Github](https://github.com/STAR-PORT/ARMOR/):

    git clone https://github.com/STAR-PORT/ARMOR.git

(This obviously requires git to be installed on your system, on Debian
and derivtives use `sudo apt-get install git`).

### Download archive

Alternatively, you can download the git repository (available from
[Github](https://github.com/STAR-PORT/ARMOR/)) as an archive and uncompress it on your computer.
You should have six files:
  - LICENCE
  - ARMOR.ipynb
  - ARMOR.py
  - Screenshot_App.png
  - README.md
  - requirements.txt

### Installing requirements (Linux)

The first requirements to be met are a working python 3 and the pip library to install dependancies.
Then you can install all the dependancies listed using in the project folder:

    pip install -r requirements.txt


### Running the code

To run the application, you have two choice:

  - launching directly with python from a terminal:
    
        python3 ARMOR.py

  - launching as a notebook:
    
        jupyter-lab ARMOR.ipynb

  inside the notebook, just press the "Restart the Kernel and run all cells" button.

A new interactive window will open with which you can play.

## 2 - Executable file (Windows only)

### Download executable file

You can download the executable file named "ARMOR.exe" from the "Release" in GitHub.

### Running the code

To run the application, double click on the executable file

    ARMOR.exe

A new interactive window will open with which you can play.
NB: the code is significantly slower with the executable (Windows) compared to the python (Linux) version
