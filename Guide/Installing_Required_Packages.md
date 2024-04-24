## Environment Setup Guide for CS777 Team Project

Welcome to our Game Recommendation System project! Before you begin, you need to make sure that all the necessary Python libraries are installed. This guide will assist you in installing these libraries via the command line.

### Prerequisites

1. **Install Python**

   Ensure that you have Python installed on your system. Our project requires Python 3.6 or higher. You can download and install it from the [official Python website](https://www.python.org/downloads/).

2. **Install pip**

   `pip` is the package manager for Python, which is used to install and manage software packages. Most Python installations already include pip. Check if pip is installed by running the following command in your command line:

   ```shell
   python -m pip --version
   ```

   If pip is not installed, you can find the installation instructions on the [official pip installation guide](https://pip.pypa.io/en/stable/installing/).

### Installing Required Python Packages

Here is a step-by-step guide on how to install each package required for your project using pip in the command line.

1. **Install PySpark**

   PySpark is the Python API for Apache Spark, which supports big data processing. Install PySpark with the following command:

   ```shell
   pip install pyspark
   ```

2. **Install Matplotlib**

   Matplotlib is a Python plotting library typically used for generating plots and graphs. Install Matplotlib with:

   ```shell
   pip install matplotlib
   ```

3. **Install TensorFlow**

   TensorFlow is an open-source machine learning library. Our project uses the TensorFlow 1.x version. Install it using:

   ```shell
   pip install tensorflow==1.15
   ```

   > Note: TensorFlow 1.15 is recommended because our project has not yet been updated to support TensorFlow 2.

4. **Install Scikit-Learn**

   Scikit-Learn is a powerful Python library for performing a variety of machine learning, preprocessing, cross-validation, and visualization algorithms. Install Scikit-Learn with:

   ```shell
   pip install scikit-learn
   ```

5. **Install Joblib**

   Joblib is used for saving and loading Python objects, which is especially useful for machine learning models. Install Joblib with:

   ```shell
   pip install joblib
   ```

6. **Install NumPy**

   NumPy is a fundamental library for scientific computing, supporting large multidimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. Install NumPy with:

   ```shell
   pip install numpy
   ```

7. **Install Pandas**

   Pandas is a library for data analysis and manipulation, offering fast, flexible, and expressive data structures. Install Pandas with:

   ```shell
   pip install pandas
   ```

8. **Install PyQt5**

   PyQt5 is a set of Python bindings for cross-platform GUI toolkit. It can be used to create desktop applications. Install PyQt5 with:

   ```shell
   pip install pyqt5
   ```

### Verifying Installation

After installing all the packages, you can verify the installation by importing them in the Python shell:

```python
import pyspark
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import joblib
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication
```

If there are no error messages, congratulations, you are ready to run the project!
