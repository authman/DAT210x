# DAT210x
Programming with Python for Data Science Microsoft

By Authman Apatira

## Current Offering

If you haven't already, [join the course](https://www.edx.org/course/programming-python-data-science-microsoft-dat210x-4)!

<center><img src='Images/py36.png' /></center>

## Getting Started

 * Download and run the [Anaconda 4.3.1 Graphical Installer](https://www.continuum.io/downloads) package for your operating system (Linux, OS/X, or Windows).
	- [Windows 64 Bit](https://repo.continuum.io/archive/Anaconda3-4.3.1-Windows-x86_64.exe)
	- [Windows 32 Bit](https://repo.continuum.io/archive/Anaconda3-4.3.1-Windows-x86.exe)
	- [macOS](https://repo.continuum.io/archive/Anaconda3-4.3.1-MacOSX-x86_64.pkg)
	- [Linux 64 Bit](https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh)
	- [Linux 32 Bit](https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86.sh)
 * Execute the following pip commands on your terminal / command line:

 ```
   sudo pip install BeautifulSoup4
   sudo pip install html5lib
   sudo pip install plyfile
 ```

 If you are on Windows (instead of Linux / Mac), you will not be able to run the `sudo` command. Try running these instead:

 ```
   python -m pip install --upgrade pip
   python -m pip install BeautifulSoup4
   python -m pip install html5lib
   python -m pip install plyfile
 ```

## What's Anaconda?

Anaconda will install `conda`, a Python, scientific package manager. It will also install a bunch of other small packages you'll use in this course:

 * matplotlib
 * numpy
 * scipy
 * pandas
 * scikit-learn
 * spyder
 * jupyter

NumPy is a popular Python module for calculating linear algebra very optimally. MatPlotLib for interactive 2D / 3D graphs, Pandas for dataset manipulation, and SciKit-Learn for machine learning.

After completing a lab exercise, submit your answers onto the appropriate course lab page in order to receive a grade. 

Spyder (Scientific PYthon Development EnviRonment) _was_ the IDE recommended you complete your lab assignments with, and you may still use it if you'd like; but for a much nicer workflow, we advise you to use Jupyter Notebooks.


## Jupyter Notebooks!

This course is in stage-one of a two part revision. As a first step, we have moved all the content from Python 2.7 to 3.6 (although it's still possible for you to program in 2.7).

We have also adopted [Jupyter Notebooks](https://jupyter.org/) (formerly IPython) and recommend you do all of your lab assignment coding in them as well. This will allot you smoothest coding experience. Some of the benefits of using notebooks: 

- Being able to document your process and keeping a (scientific) record of everything you tried while experimenting.
- Th ability to add blocks of code, the same way you would in a flat file--but have them execute immediately if you like, the same way they would in Python shell.
- Markdown support, so you can interlace comments and notes directly (and aesthetically) right into your code.
- Ability to import external scripts.
- Spell checker and thousands of other plugins.
- Secure remote access via a built-in web server, so you can run code on a remote box.
- You can get immediate help on commands or functions that you aren't familiar with using special keywords like `?? method_name`.
- Etc.

Jupyter comes with Anaconda. To start it up, simply navigate to the class repository folder you downloaded (the folder that contains this file), and then type the following command into your console:

    $ jupyter notebook
    
This will launch a new Jupyter instance in your web browser, pointed at your current working directory. From there, you can open the lab `.ipynb` Interactive Python Notebook files, or even create new ones of your own. Each notebook you launch will open up and run in its own, Python interpreter sandbox. If you have multiple versions of Python installed, you can specify from the menu what version of Python you'd like to spin up to interpret your code with.

### Basic Jupyter Commands ##

- Execute the code in a cell by hitting **Shift+Return**.
- Press the **y** key to convert a cell to code.
- Press the **m** key to convert a cell to markdown.
- To install notebook extensions, type the following commands and then relaunch jupyter from your shell:   
	`conda install -c conda-forge jupyter_contrib_nbextensions`   
	`jupyter contrib nbextension install --user`   
	
After enabling notebook extensions, you can browse them from the (now visible) NBExtensions tab.

If you'd like additional help getting started with Jupyter notebooks, please [visit this page](https://www.youtube.com/results?search_query=jupyter+notebook+tutorial) for some short, getting-started, walk through videos.

Good luck!
