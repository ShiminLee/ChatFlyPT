# Getting started

> The White Rabbit put on his spectacles. 'Where shall I begin, please your Majesty?' he asked. 'Begin at the beginning,' the King said gravely, 'and go on till you come to the end: then stop.' (Lewis Carroll)

In this post I will follow the White Rabbit and attempt to describe how to go from a computer with no special software installed to a first successful connection to the Connectome data in Python. This means, you will have a working Python-3 and a virtual environment with the project's dependencies (defined in [`pyproject.toml`](../pyproject.toml)) installed, and the neuPrint authentication configured — verified by a short connection test at the end of this document. If you don't know what this means or want to confirm that everyone is on the same page then the rest of the document is intended to help you set everything up. Otherwise, if you already have a working setup, feel free to skip the rest of the document and continue with [`run_order.md`](run_order.md), which describes the data pipeline and figure notebooks.

## What version of Python?

The short answer: If you are on a Linux-like system, you will most likely have the latest version of _vanilla_ Python installed.

For MacOS I recommend using Miniconda, a version of Anaconda but without unnecessary software bloat.

For Windows, _vanilla_ or Miniconda Python should be similarly easy to install and use.

The _vanilla_ version will usually be more up to date and recent than any other derived distribution like *conda or Active Python. With their focus on usability, some of the other distributions might be easier to use, depending on your preferences and experience.

## Virtual environments in Python

Python libraries and applications are usually distributed in so called packages. These packages can be installed through Python itself, independently of the operating system. Packages can depend on each other and sometimes these dependencies are for very specific versions of a library or package. If you work (or plan to work) on different python projects with different libraries, these dependencies can be in conflict with each other. Python supports a mechanism to use different versions of a library for different software projects on the same computer. To do this software projects and their dependencies are organized in _virtual environments_.

There are several ways to manage virtual environments. For _vanilla_ Python (on Linux and Windows) I recommend a software called [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), for Miniconda or Anaconda (on MacOS and Windows) I recommend using the built in [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage environments.

The _vanilla_ Python comes with a software package manager called `pip` (_p_ackage _i_nstaller for _P_ython). The same software package manager is also available inside *conda environments. In addition *conda allows the installation of packages through the `conda` package manager.

### Install Python 3 Linux

Most likely Python 3 is already installed. Just open a terminal, type `python --version` and as long as the reponse contains a version bigger than `Python 3.5.x` you should be fine.

If not and depending on your linux distribution, a `sudo apt update; sudo install python3` (on Debian / Ubuntu) or `pacman -S python` (Arch Linux) will install the latest version.

#### Enable virtual environments in Linux and _vanilla_ Python

The software [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) is helpful in managing virtual environments. To install, use `sudo apt update; sudo install virtualenvwrapper` (on Debian derivates) or `pacman -S python-virtualenvwrapper` (Arch Linux).

Once installed, you can create an environment with `mkvirtualenv <NAME>`. Change into environments with `workon <NAME>` and deactivate an environment with `deactivate`.

### Install Python 3 on MacOS

The Python version that comes with MacOS is too old and would not work with the connectomics data. Instead of updating the system Python, I recommend installing Miniconda, a version of Anaconda:

1. Download the miniconda `*.pkg` file from the [website](https://docs.conda.io/en/latest/miniconda.html). Choose either the Intel x86 or the Apple M1 package, depending on your processor. 
2. Click on the downloaded `*.pkg` file to start the installer and confirm the licenses and directories.
3. Open a new terminal window. If you had a terminal open, close it first and then open another one. You can confirm that your installation of Miniconda worked by typing `conda list` – that should give you some output other than an error message.
4. For easy GUI access to the Anaconda environments, install the anaconda navigator by running `conda install anaconda-navigator` in a terminal.

#### Enable virtual environments in MacOS and _*conda_ Python

There are two ways to create a virtual environment in Miniconda and Anaconda: either through the anaconda-navigator or command line. In the anaconda-navigator, click on the _Environment_ tab, then the _Create_ button and provide a `<NAME>` for the environment. To enter an existing environment from the anaconda-navigator, click on the white triangle on green circle next to the environment's name and launch a terminal from there.

Alternatively you can create an environment with `conda create --name <NAME>`. Change into the environment with `conda activate <NAME>` and deactivate it with `conda deactivate`.

### Alternative to Anaconda on MacOS

In case you prefer not to use Anaconda, these steps will guide you in creating a local environment on MacOS.

##### Install Python 3 without Miniconda
Note: If Python 3 is already installed and functional, skip this section. If you need to update Python to a newer version, follow steps 5–6 only.

MacOS typically comes with Python pre-installed, but often it's an outdated version that is no longer supported. This version may not be compatible with the libraries and tools required for accessing connectomics data. To check your current Python version, open a terminal window (Finder → Applications → Utilities → Terminal) and type `python --version`. IIf the command returns a version newer than `Python 3.5.x` you might be fine. Otherwise, you will need to upgrade your Python version.

The article [The right and wrong way to set Python 3 as default on a Mac](https://opensource.com/article/19/5/python-3-default-mac) provides a good setup guide. While this article includes a link to [download Python from python.org](https://opensource.com/article/19/5/python-3-default-mac#use-python-3-as-the-macos-default), we recommend a different approach. Here are the steps:

1. Open a terminal window.
2. Install Command line tools for [Xcode](https://apps.apple.com/us/app/xcode/id497799835) with `xcode-select --install`.
3. Install [Homebrew](https://docs.brew.sh/) using `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"`.
4. Install build dependencies with `brew install openssl readline sqlite3 xz zlib tcl-tk` (see <https://github.com/pyenv/pyenv/wiki#suggested-build-environment>)
5. Install [pyenv](https://github.com/pyenv/pyenv) with `brew install pyenv`.
6. Install the latest Python version, for example by typing `pyenv install 3.11.6` (check the latest version with `pyenv install -l`).
7. Set the global Python version with `pyenv global 3.11.6`.
8. Update your zsh environment: `echo -e 'eval "$(pyenv init -)"' >> ~/.zshrc`.
9. Update your bash environment: `echo -e 'eval "$(pyenv init -)"' >> ~/.bash_profile`.
10. Close your terminal.
11. Open a new terminal window (Finder → Applications → Utilities → Terminal) and verify your Python version with `python --version`.

#### Set up the environment

1) Open a terminal window
2) Ensure git is installed. If not, use Homebrew (refer to previous 'Install Python 3 without Miniconda', step 3): 
	a) Install [Homebrew](https://docs.brew.sh/) via `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"`
	b) Install git `brew install git
3) Navigate to your optic lobe connectome directory (via `cd 'your directory')
4) Create a new environment via `python -m venv .venv`. This creates a virtual environment in the directory `.venv` in the current location.
5) Activate the environment by using the activation script inside the new environment: `source .venv/bin/activate`. Repeat this step whenever you need to re-enter the environment.
6) Install dependencies with `pip install -e .`

#### Updating the dependencies

To update dependencies, 1) activate your environment with `source .venv/bin/activate` and 2) run `pip install -e .` again. If this fails, delete the `.venv` environment from your directory and repeat steps 3-6 from 'Set up the Environment'.

### Install _vanilla_ Python 3 on Windows

On Windows it should be similarly easy to use either a _vanilla_ version of Python 3 or the Anaconda version.

To install the _vanilla_ version of Python 3, head over to the [official download page](https://www.python.org/downloads/). Download the installer, execute it, and confirm the choices for the installation.

Open a terminal (eg right-click on start → Windows Terminal) and check your Python version with `python --version`.

#### Enable virtual environments in Windows and _vanilla_ Python

Similarly to the Linux _vanilla_ Python environment, I recommend virtualenvwrapper to manage virtual environments in _vanilla_ Python under Windows. Once Python is installed, you can install the [virtualenvwrapper-win](https://pypi.org/project/virtualenvwrapper-win/) package for Windows via `pip install virtualenvwrapper-win`. To see if it worked, type `lsvirtualenv` in your terminal – this should return an empty list, not an error. If it doesn't work immediately, try to close and reopen your terminal window or try _turning it off and on againⓇ_.

The package above works for the normal windows terminal. If you are using the Powershell, please install [virtualenvwrapper-powershell](https://pypi.python.org/pypi/virtualenvwrapper-powershell) instead.

To create a virtual environment use `mkvirtualenv <NAME>`. Change into the environment with `workon <NAME>` and leave the environment with `deactivate`.

### Install _Miniconda_ Python 3 on Windows

If you prefer to use the Anaconda version, download the latest Miniconda from the [website](https://docs.conda.io/en/latest/miniconda.html). Install the package and open a new terminal. Check if the installation worked with `conda list`. Similarly to the MacOS version, I recommend to install the anaconda navigator through `conda install anaconda-navigator`.

#### Enable virtual environments in Windows and _*conda_ Python

The procedure to enable virtual environments inside the _*conda_ environments under Windows is exactly the same as under MacOS. Please follow the instructions there on how to do it.

## Create the environment

With the virtualenvwrapper or conda installed, creating a new virtual environment is as easy as running the command `mkvirtualenv <NAME>` (or `conda create --name <NAME>`) in your terminal, where `<NAME>` is any name you choose. This will create the environment and automatically activate it. To end using a virtual environment, type the command `deactivate`. To start working within an existing environment, run the command `workon <NAME>` to enable it again.

## Extra step for Windows

We used python 3.10 on our Windows machines.

One of our libraries requries a C++ compiler. While MacOS and Linux have these preinstalled, you will need to do an extra step if you are using Microsoft Windows. Go to the [Microsoft Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) website and [download and run the build tools installer](https://aka.ms/vs/17/release/vs_BuildTools.exe). In the _Workloads_ tab, select at least _Visual C++ build tools_, and in the _Individual components_ tab, select at least _Windows 10 SDK_ or _Windows 11 SDK_, _Visual C++ tools for CMake_, _Testing tools core features - Build Tools_, _C++/CLI support_,  and _VC++ 2015.3 v14.00 (v140) toolset for desktop_. Once the installation is finished, you should be able to continue with the [dependency management](#dependency-management) below.

## Dependency management

From this step forward you will need to have the code from the [visualpathways](https://github.com/reiserlab/visualpathways) github project. So please download the code to a directory, which I will call _project root_ (or `$PROJECT_ROOT`) from now on.

Open a terminal and go to the _project root_. The project's dependencies are defined in `pyproject.toml`. To install everything, first make sure that you have the virtual environment activated, for example by executing `workon <NAME>` (or `conda activate <NAME>`). If you are not sure, it doesn't hurt to run the command again.

Now you can install all packages via the command `pip install -e .`. Make sure that you did the [extra step in Windows](#extra-step-for-windows). If you run this the first time, it might take a few minutes to download all required libraries. This will include some basic tools, but also the [neuprint-python](https://github.com/connectome-neuprint/neuprint-python) library that is required to access the connectome data. (For an exactly-reproducible environment with pinned versions, use `pip install -r requirements-lock.txt` instead.)

To use the environment in Jupyter notebooks, register it as a kernel by running `python -m ipykernel install --user --name <a-name-of-your-choice>` in a terminal that still has the virtual environment enabled, then select that kernel when opening a notebook. The kernel name is entirely up to you — the kernel names stored in the notebooks' metadata are just whatever the last editor happened to use, and nothing in this project depends on them.

To update the packages in the environment, activate it and run `pip install -e .` again. 

## Project configuration

Access to the connectome data via neuprint requires authentication. Create a copy of the `.env_sample` file with the name `.env` inside your _project root_. To add your own credentials, you will need to go to the account page at [neuPrint](https://neuprint-cns.janelia.org/account) and copy the (very long) AuthToken to the line starting with `NEUPRINT_APPLICATION_CREDENTIALS=`. The other variables (server URL, dataset name, mesh/ROI sources) are pre-filled with the correct values — double check that `NEUPRINT_DATASET_NAME=male-cns:v1.0` is set.

Setting the plotting backend for navis via the environment is convenient as well. For example, to use the plotly library, add this:

```sh
# Configuration of a Jupyter backend for 3D plots
NAVIS_JUPYTER_PLOT3D_BACKEND=plotly
```

## Verify the connection

Now you should be able to verify your setup with a short connection test. From the _project root_, run:

```shell
python -c "
from pathlib import Path
import sys
from dotenv import find_dotenv
sys.path.append(str(Path(find_dotenv()).parent / 'src'))
from utils import olc_client
olc_client.connect(verbose=True)
"
```

Within a few seconds you should see a confirmation that lists the server, the dataset, and the account associated with your token — something like:

```shell
Connected to https://neuprint-cns.janelia.org[male-cns:v1.0].
Client: neuprint-python v1.7.3
User: <your-email>@janelia.hhmi.org [readwrite]
```

If this works, your environment and credentials are complete. What to run next — the `setup_data` cache-building stages and the figure notebooks — is described in [`run_order.md`](run_order.md). Note that the figure notebooks read from a local cache that has to be built first; the cache build hits the neuPrint server and some stages are slow.

## File structure

You might have noticed that the folder `$PROJECT_ROOT/src/setup_data` contains two files starting with `s1_malecns_data`: one python file with the extension `.py` and one Jupyter notebook file with the extension `.ipynb` (the same is true for the other numbered stages there and the notebooks in `src/make_figures`). These two files are connected: if the Python file is edited, this change will automatically be pulled into the notebook the next time it is opened. Also, if edits are done inside the notebook file, the changes will be saved to the Python file.

The connection is done through the [JupyText](https://jupytext.readthedocs.io) software which combines the best of at least three worlds: 1) easy tracking of changes in text files, 2) saving input and output in the notebook format, 3) execution of whole scripts without complicated environments. Here I just wanted to explain why there are two files with the same name and similar extensions in these folders.

## Final words

If any of the above failed and you can't fix it, then please open an issue on the [project's GitHub page](https://github.com/reiserlab/visualpathways/issues) or get in contact with the Reiser Lab.
