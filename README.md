ACTABLE AI
=======

Git Submodules Guide
--------------------

Full documentation available here: https://git-scm.com/book/en/v2/Git-Tools-Submodules

Git Submodules is a feature used to contain the autogluon library directly in the working directory at specific commit.
When cloning for the first time the repository this command should be executed:
```shell
git submodule update --init --recursive
```

Then when pulling new changes you can either run the command above after pulling or directly use this command:
```shell
git pull --recurse-submodules
```

Finally when the submodule content needs to be updated to a specific commit the following commands must be executed:
```shell
cd third_parties/autogluon
git fetch --all
git checkout <commit_hash>
cd ../..
git add third_parties/autogluon
```


**Prerequisite installing for actableai library.**  

0. Environment
- ubuntu 18.04
- python 3.7

1. Install R
- sudo apt update -qq
- sudo apt install --no-install-recommends software-properties-common dirmngr
- sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
- sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
- sudo apt install --no-install-recommends r-base r-base-dev libbz2-dev
- sudo R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'  
  
2. Install dependences
- pip install pip==19.3
- pip install rpy2==2.9.6b0

3. Install actableai
- pip install -r requirements.txt



**Note for MacOS Users**

To install R you can use these commands (assuming you have brew installed):
- brew install r
- R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'

You will also need to install OpenMP:
- brew install libomp
