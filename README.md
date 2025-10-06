# Installing and running ARMOR

## 0 - Requirements

This software requires python3  and the following libraries:
  - math: https://docs.python.org/3/library/math.html
  - numpy: https://numpy.org/
  - matplotlib: https://matplotlib.org/stable/index.html
  - scipy: https://scipy.org/install/
  - tkinter: https://docs.python.org/3/library/tkinter.html
  - (optional but recommanded for Linux) pip: https://docs.python.org/3/installing/index.html
  - (optional in Linux) jupyter-notebook: https://jupyter.org/

## 1 - Linux/Ubuntu

### Download source code with Git
The source code is available from
[Github](https://github.com/STAR-PORT/ARMOR/):

    git clone https://github.com/STAR-PORT/ARMOR.git

(This obviously requires git to be installed on your system, on Debian
and derivtives use `sudo apt-get install git`).

### Download archive

Alternatively, you can download the git repository (available from
[Github](https://github.com/STAR-PORT/ARMOR/)) as an archive and uncompress it on your computer.
You should have four files:
  - LICENCE
  - ARMOR.ipynb
  - ARMOR.py
  - README.md

### Installing requirements

The first requirement to be met is a working python 3. You can install it using the following command line:

    sudo apt-get install python3.X

replace X by the lastest version (check https://www.python.org/downloads/).
Note that in Ubuntu and probably other Linux based OS python is already install.

For installing the libraries, we recommand using pip.
You can install it using the following command line:

    sudo apt-get install python3-pip

Then you can install all the libraries listed using:

    pip install <librariename>

replacing librariename by the name of the librarie (without the <>).

### Running the code

To run the application, you have two choice:

  - launching directly with python from a terminal:
    
        python3 ARMOR.py

  - launching as a notebook:
    
        jupyter-notebook ARMOR.ipynb

  inside the notebook, just press the "Restart the Kernel and run all cells" button.

A new interactive window will open with which you can play.

## 2 - Windows/MacOS

### Download archive

You can download the git repository (available from
[Github](https://github.com/STAR-PORT/ARMOR/)) as an archive and uncompress it on your computer.
You should have four files:
  - LICENCE
  - ARMOR.ipynb
  - ARMOR.py
  - README.md

### Installing requirements


### Running the code

To run the application :

inside the notebook, just press the "Restart the Kernel and run all cells" button.

A new interactive window will open with which you can play.
