
# Repository of  VulSeeker
## I. Introduction of VulSeeker
It's a semantic learning based vulnerability search tool for cross-platform binary. Given a vulnerability function `f`, VulSeeker can identify whether a binary program contains the same vulnerability as `f`. Currently, it support six architectures, such as X86, X64, ARM32, ARM64, MIPS32, MIPS64. If you meet any problems, please feel free to email me at gaojian094@gmail.com.

## II. Prerequisites
To use VulSeeker, we need the following tools installed
- IDA Pro - for generating the LSFG （data flow graph and control flow graph）and extracting features of basic blocks
- python2.7 - all the source code is written in python2.7 
- [miasm](https://github.com/cea-sec/miasm) - for converting assembly program to LLVM IR. We extend it to support more assembly instructions. Please directly copy the `miasm2` provided by us to the python directory of `IDA Pro`.

## III. Directory structure
- `0_Libs/search_program`: it contains the binary file considered as the target from which VulSeeker search vulnerability.
- `1_Features/search_program`: it contains the instruction features, control flow graph and data flow graph for each function in the target. 
- `4_Model/VulSeeker`: it is a DNN model that we have trained to use directly.
- `5_CVE_Feature`: It contains the instruction features, control flow graph and data flow graph of each version of the two vulnerabilities (CVE-2014-3508, CVE-2015-1791).
- `6_Search_TFRecord`: Tfrecord data file is a binary file that stores data and labels in a unified way. It can make better use of memory and make rapid replication, movement, reading and storage in tensorflow. 
- `7_Search_Result`: All the search result list will be stored here.

## IV. Usage
1. We need modify the `config.py` file. All the dependency directories can be modified here. Simple modification is listed as following, but it need to follow the directory structure we defined:
```
IDA32_DIR = "installation directory of 32-bit IDA Pro program"
IDA64_DIR = "installation directory of 64-bit IDA Pro program"
```
2. We put the programs to be searched in the `VulSeeker/0_Libs/search_program` directory.
3. We run the `VulSeeker/command.py` file to generate the labeled semantic flow graphs and extract initial numerical vectors for basic blocks. The result files should be placed in the `1_Features/search_program` directory.
4. We execute the `VulSeeker/search_by_list_vulseeker.py` file to obtain embedding vectors of the functions and get the function list in descending order of similarity scores. 

` Note: All steps can be executed in the Linux system.`

## V. Viewing the search result
The following figure is an example of the search result.

![avatar](./fig/search_example.png)

 For each vulnerability function, there are a total of 48 compiled versions. These versions contain different architectures (X86, X64, ARM32, ARM64, MIPS32 and MIPS64), compilers (GCC v4.9 and GCC v5.5) with four optimization levels (O0-O3). 
- Column A records the function name.
- Column B is the average similarity score between the corresponding function and the vulnerability function with 48 compiled versions.
- Column C records the file to which the function belongs.
- The other items after column C are the similarity scores between a particular version of the vulnerability and the corresponding function.


## VI. Build VulSeeker from source code for model modification and retraining
### Optional installation and configuration: Python-2.7.13
If you have an appropriate Python-2.7 version, you can skip this installation. Please make sure that you have installed Python with ucs4 unicode encoding. You can identify ucs2 and ucs4 with the following code.
```
>> import sys
>>print sys.maxunicode
1114111# it means the ucs4 encoding
65535# it means the ucs2 encoding, you need reinstall your python. The tensorflow-1.1.0 requires the ucs4 unicode encoding style.
```
1. install required libraries, or it will cause some troubles.
`sudo apt-get install python-dev libffi-dev libssl-dev libxml2-dev libxslt-dev libmysqlclient-dev libsqlite3-dev zlib1g-dev libgdbm-dev`
2. download and install Python-2.7.13
`wget -c https://www.python.org/ftp/python/2.7.13/Python-2.7.13.tar.xz
xz -d Python-2.7.13.tar.xz 
tar xf Python-2.7.13.tar 
cd Python-2.7.13
./configure --prefix=/usr/local/python2713 --enable-unicode=ucs4
make
make install
`
3. install setuptools and pip package
` wget https://bootstrap.pypa.io/ez_setup.py -O - | sudo python
curl -O https://bootstrap.pypa.io/get-pip.py
python get-pip.py
`
4. link pip and python to bin path
`rm /usr/bin/pip2
rm /usr/bin/pip2
ln -s /usr/local/python2713/bin/pip /usr/bin/pip2
ln -s /usr/local/python2713/bin/pip /usr/bin/pip
rm /usr/bin/python
rm /usr/bin/python2
ln -s /usr/local/python2713/bin/python /usr/bin/python2
ln -s /usr/local/python2713/bin/python /usr/bin/python
`
5. add environment variables 
`export PATH="$PATH:/usr/local/python2713/lib/python2.7/site-packages:/usr/local/python2713/bin"
`

### Required installation

If you want to train your own network model, you need to install tensorflow-1.1.0 version. We build this version of tensorflow from source code. The following is the detailed installation instructions (for cpu-only tensorflow) on the ubuntu14 machine.

1. install dependent packages
```
sudo apt-get install zlib1g-dev swig python-wheel pkg-config zip g++ unzip python-numpy python-dev
wget https://pypi.python.org/packages/c8/0a/b6723e1bc4c516cb687841499455a8505b44607ab535be01091c0f24f079/six-1.10.0-py2.py3-none-any.whl#md5=3ab558cf5d4f7a72611d59a81a315dc8 #download and install six
sudo pip install six-1.10.0-py2.py3-none-any.whl 
sudo pip install networkx
sudo pip install pyparsing
sudo pip install numpy
```
2. install bazel building tool
- Download `bazel-0.4.2-installer-linux-x86_64.sh` from https://github.com/bazelbuild/bazel .
- `chmod +x bazel-0.4.2-installer-linux-x86_64.sh`
- `./bazel-0.5.4-installer-linux-x86_64.sh --user`
- add bazel file path to the PATH environment variable. e,g,: `export PATH="$PATH:$HOME/bin"`
3. install java8/openjdk8
`sudo add-apt-repository ppa:openjdk-r/ppa
 sudo apt-get update
 sudo apt-get install openjdk-8-jdk
 sudo update-alternatives --config java    #note: select the appropriate version
 sudo update-alternatives --config javac`
4. install tensorflow
- $git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git -b r1.1   #download source code,--recurse-submodules is used for downloading the dependent tools，-b r1.1 means the tensorflow-1.1.0 version.
-  enter the tensorflow directory and then select the python path
e.g.,`./condigure /usr/bin/python`
```
note: the following is the selection during the installation process.
malloc implementation: Y
Google Cloud Platform support: N
Hadoop File System support: N
XLA just-in-time compiler: N
Python library paths: Default is [/usr/local/lib/python2.7/dist-packages],you can select a different path.
OpenCL support: N
CUDA support: N
Configuration finished.
```
- execute `bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package` to build the tensorflow source code
- execute `./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg` to get the installation wheel tensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl
- install the tensorflow package `sudo pip install /tmp/tensorflow_pkgtensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl`, it will also install funcsigs mock pbr protobuf.
- verify the installation
```
$python
>>import tensorflow as tf
>>hello=tf.constant('Hello, tensorflow!')
>>sees=tf.Session()
>>print sees.run(hello)
Hello, tensorflows!
>>a=tf.constant(10)
>>b=tf.constant(32)
>>print sees.run(a+b)
42
```
### Usage
It is consistent with the usage described above.

