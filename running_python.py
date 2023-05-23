# -*- coding: utf-8 -*-
"""running_python.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/astg606/py_materials/blob/master/welcome/running_python.ipynb

<center>
<table>
  <tr>
    <td><img src="http://www.nasa.gov/sites/all/themes/custom/nasatwo/images/nasa-logo.svg" width="100"/> </td>
     <td><img src="https://github.com/astg606/py_materials/blob/master/logos/ASTG_logo.png?raw=true" width="80"/> </td>
     <td> <img src="https://www.nccs.nasa.gov/sites/default/files/NCCS_Logo_0.png" width="130"/> </td>
    </tr>
</table>
</center>

        
<center>
<h1><font color= "blue" size="+3">ASTG Python Courses</font></h1>
</center>


---

<center><h1>
    <font color="red" size="+2">Running Python</font>  
</h1></center>
"""

import numpy as np
arr = np.array([1, 2, 3, 4])

reshaped_arr = arr.reshape(2, 2)
print(reshaped_arr)

arr = np.array([1, 2, 3])

mean = np.mean(arr)
print(mean)

import pandas as pd

data = {'Name': ['John', 'Jane', 'Mike'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Paris', 'London']}

df = pd.DataFrame(data)
print(df)

name_column = df['Name']
print(name_column)
filtered_df = df[df['Age'] > 30]
print(filtered_df)
sorted_df = df.sort_values('Age', ascending=False)
print(sorted_df)

df.drop_duplicates()

df = pd.read_excel('data.xlsx')

"""# <font color='blue'>Why Learn Python?</font>

---

* Is a free and open-source.
* Is a high-level and interpreted general purpose programming language.
* Its simple syntax makes it suitable for learning programming as a first language.
* Has a very extensive standard library and over 150,000 external libraries.
* Is portable and extensible. It interoprate with other languages such as Fortran, C, C++, Java,...
* Has a large community.

**Benefits of Learning Python**

![fig_python](https://static.javatpoint.com/interview/images/advantages-of-python.png)
Image Source: www.javatpoint.com

Python finds applications in areas such as:

+ Web Programming
+ Data Science
+ Machine Learning and Artificial Intelligence
+ Scripting & Automation
+ Games
+ Desktop GUI applications
+ Database access
          
![fig_python](https://hackernoon.com/hn-images/1*jVKTE1dd8CVv4WEtcidCPA.png)

# <font color='blue'> Python Versions</font>

----

Python has two major versions:

+ **2.x**: Released in 2000. The latest version is 2.7 released in 2010. It is not recommended for use in new projects.
+ **3.x**: Released in 2008 to fix problems that exist in Python 2.x. 
   - The nature of these changes is such that Python 3 was incompatible with Python 2. It is backward incompatible.
   - Python 3 isn’t compatible with Python 2. 
   - You should use the latest versions of Python 3 for your new projects.

**For this class, we will use Python 3.x.**

# <font color='blue'>Four Ways to Run Python</font>

---

* Interactive Command Line through `python`
* Interactive Command Line through `ipython`
* Scripting
* Jupyter Notebook

You first need to make sure that Python is install on your system. 
With Linux/Unix systems, a default Python interpreter comes with the operating system. To test this, from the terminal type:

`which python`

and you will see something like:

`/usr/bin/python`

### <font color='red'> Interactive Command Line through `python` </font>

---

* You can use the interpreter in the interactive mode to test some commands.
* You need to type `python` from the command line.
* You can directly type in Python code, and press `Enter` to get the output.
* You can exit the interactive mode with `quit()` or `exit` command or `^Ctrl + D`.
* The sequence you enter will not be saved if you close the current session.

**Sample Session**

1. Open your terminal/Anaconda prompt and type:

`python`


2. From this interactive shell, you will notice that the prompt is `>>>`. This is now allowing us to type Python code directly and execute it.

3. Now type:

`print('Hello world!')`

4. You should see the output on the screen. 

5. To exit the Python shell type `exit()` and press `Return` key.

### <font color='red'>Interactive Command Line through `ipython` </font>

---
* IPython is an interactive shell for the Python programming language that offers enhanced introspection, additional shell syntax, tab completion and rich history.
* It does not come by default with Python.
* IPython gives you all that you get in the basic interpreter but with a lot extra (line numbers, advanced editing, more functions, help functions etc).
* It can be started by typing `ipython` at the command line.
* The main aesthetic difference between the Python interpreter and the enhanced IPython interpreter lies in the command prompt: Python uses `>>>` by default, while IPython uses numbered commands (e.g. `In [1]:`).

**Sample Session**

1. Open your terminal/Anaconda prompt and type:

`ipython`

2. This is an enhanced interactive shell that has many features (tab-completion, woot!). It also has a prompt that is numbered.

3. Now type:

`print('Hello world!')`

4. You should see the output on the screen.

5. To exit the Python shell type `exit()` and press `Return` key.

### <font color='red'> Scripting </font>

---
* Real Python programs are made as scripts and look like simple text files. 
* These files are given extensions `.py`.
* You can create text files using whatever text editor you like.
* To run the script you need to use the programming language interpreter and specify the name of the created file as an additional parameter:

```
       %python my_python_file.py
```

**Sample Session**

1. Start a new ASCII/text document named `helloworld.py` and enter the following text:

`print('Hello world!')`
 
2. Save the document and then in your terminal/Anaconda prompt, go to the directory containing the newlt created file.


3. From the terminal/Anaconda prompt, type:

`python helloworld.py`

4. You should see the output on the screen.

### <font color='red'> Jupyter Notebook </font>

---
* A useful hybrid of the interactive terminal and the self-contained script is the Jupyter notebook, a document format that allows executable code, formatted text, graphics, and even interactive features to be combined into a single document. 
* Though the notebook began as a Python-only format, it has since been made compatible with a large number of programming languages. 
* The notebook is useful both as a development environment, and as a means of sharing work via rich computational and data-driven narratives that mix together code, figures, data, and text.

**Sample Session**

1. Open your terminal/Anaconda Prompt and type:

`jupyter notebook`
 
2. This directs you to a web browser and you can navigate to an already existing notebook or create one (right side menu New -> Python 3).

3. This will open up a new Untitled notebook where you can directly input Python code, Markup formatted text, or have raw text.

4. Now type:

`print('Hello world!')`

5. Press `Shift+Enter`, `Cntrl+Enter` or click `Cells -> Run Cells` or use the `Play` button near the top of the page.

6. You should see the output on the screen. 

7. Exit via closing the browser windows and stopping the server running in the terminal/command prompt (most likely with a Cntrl+C).

## <font color='red'>Other Ways</font>

* [10 Best Python IDE & Code Editors](https://hackr.io/blog/best-python-ide)
* [Python IDEs and Code Editors (Guide)](https://realpython.com/python-ides-code-editors-guide/)
* [Google Colaboratory](http://colab.research.google.com)
* [Binder](http://mybinder.org)
* [Microsoft Azure](http://notebooks.azure.com)

# <font color='blue'>References</font>

* [How to Run Python Code](https://jakevdp.github.io/WhirlwindTourOfPython/01-how-to-run-python-code.html)
* [How To Run Your Python Scripts](https://www.knowledgehut.com/blog/programming/run-python-scripts)
* [Getting Started with Python in Visual Studio Code](https://code.visualstudio.com/docs/python/python-tutorial)
* [How to Run a Python Script on Mac](https://www.maketecheasier.com/run-python-script-in-mac/)
* [Running Python Scripts from anywhere under Windows](https://correlated.kayako.com/article/40-running-python-scripts-from-anywhere-under-windows)
"""