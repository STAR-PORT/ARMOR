# Installing and running ARMOR

## 0 - Requirements

This software requires python3  and the following libraries:
  - math: https://docs.python.org/3/library/math.html
  - numpy: https://numpy.org/
  - matplotlib: https://matplotlib.org/stable/index.html
  - scipy: https://scipy.org/install/
  - tkinter: https://docs.python.org/3/library/tkinter.html
  - (optional but recommanded for Linux) pip: https://docs.python.org/3/installing/index.html
  - (optional in Linux) jupyterlab: https://jupyter.org/install

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
  - ARMOR.exe
  - Screenshot_App.png
  - README.md

### Installing requirements (Linux)

The first requirement to be met is a working python 3. You can install it using the following command line:

    sudo apt-get install python3.X

replace X by the lastest version (check https://www.python.org/downloads/).
Note that in Ubuntu and probably other Linux based OS python is already install.

To install the libraries, we recommand using pip.
You can install it using the following command line:

    sudo apt-get install python3-pip

Then you can install all the libraries listed using:

    pip install <libraryName>

replacing libraryName by the name of the librarie (without the <>).

### Installing requirements (Windows)

1. Go to the official Python website: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. Download the latest **Python 3.x** installer.
3. Run the installer and **make sure to check** the box:
   Add Python 3.x to PATH
4. Click **Install Now** and wait until the installation completes.

To Verify the installation:
Open Command Prompt and type:

    python --version
    pip --version

You should see the installed Python and pip versions.

In Command Prompt, navigate to the project folder and run:

    pip install <libraryName>

replacing libraryName by the name of the librarie (without the <>).

### Installing requirements (MacOs)

Most macOS versions already include Python 3. To verify, open Terminal:

    python3 --version
    pip3 --version

If Python is not installed or outdated, download the latest Python installer for macOS: https://www.python.org/downloads/macos/

Run the installer and follow the instructions.

Navigate to the project folder in Terminal and run:

    pip install <libraryName>

replacing libraryName by the name of the librarie (without the <>).
  

### Running the code

To run the application, you have two choice:

  - launching directly with python from a terminal:
    
        python3 ARMOR.py

  - launching as a notebook:
    
        jupyter-lab ARMOR.ipynb

  inside the notebook, just press the "Restart the Kernel and run all cells" button.

A new interactive window will open with which you can play.

## 2 - Executable file (Windows only)

### Download archive

You can download the git repository (available from
[Github](https://github.com/STAR-PORT/ARMOR/)) as an archive and uncompress it on your computer.
You should have six files:
  - LICENCE
  - ARMOR.ipynb
  - ARMOR.py
  - ARMOR.exe
  - Screenshot_App.png
  - README.md


### Running the code

To run the application, double click on the executable file

    ARMOR.exe

A new interactive window will open with which you can play.
NB: the code is significantly slower with the executable (Windows) compared to the python (Linux) version
