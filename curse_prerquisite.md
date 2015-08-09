##Introduction

These are the prerequisites for the Deep Learning Tutorial. There are maybe as many deep learning frameworks as JavaScript frameworks, but we will concentrate on the Python Package [nolearn](https://github.com/dnouri/nolearn), which was already sucessfully used for the last DSR batch and which contains a number of wrappers and abstractions around existing neural network libraries, most notably [Lasagne(Theano)](https://github.com/Lasagne/Lasagne). 

You can install everything locally on your Mac, but I recommend to use a ec2 gpu instance, especially if you don't have a Nvidia GPU in your machine. I tested the code on the AMI Theano - CUDA 7 (ami-b141a2f5) (only US West (N. California)), which is the updated version of the one introduced in this blog post [Installing Theano on AWS](http://markus.com/install-theano-on-aws/). Please make sure to only get the small GPU instance and only as a spot instance to keep the costs low.  

Just for completeness: you don't need to use a machine with a Nvidia gpu, but I heavily recommend it.

##Platform specific prerequisite

###Local OSX installation

I assume that python, pip, scipy, numpy and ipython notebook are already installed.

To install Cuda download the installer from: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

After the installation set your environment variables as described in the documentation as follows:

`export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-7.0/lib:$DYLD_LIBRARY_PATH `

`export PATH=/Developer/NVIDIA/CUDA-7.0/bin:$PATH`

Test that Cuda is successfully installed with following command:

`nvcc -V`

The most common error is, that after a successful installation the environment paths are not correctly set.  

###AWS Instance (Ubuntu 14.10)

Cuda is already installed and configured but ipython notebook and packages for matplotlib are missing:

`pip install ipython[notebook] --user --upgrade   # installing ipython notebook`

`sudo apt-get install libpng-dev libfreetype6-dev  # for matplotlib`

##General prerequisite

Afterwards install python packages necessary for the tutorial with:

`#  installing Theano and lasagne, sklearn and other packages for nolearn  `

`pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt --user `

`# installing nolearn`

`pip install git+https://github.com/dnouri/nolearn.git@3fdd21d819#egg=nolearn --user`

At the end download and unzip the data from the[ kaggle facial keypoint detection challange](https://www.kaggle.com/c/facial-keypoints-detection/data)

#Testing

To test that the installed packages are working enter  

`THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python import -c 'nolearn, theano'`

which should output the Cuda device that will be used.  

#Remarks

We will use ipython notebook for all of our coding. Please make sure to always start ipython notebook like:

`THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ipython notebook`

Otherwise Theano will use the cpu and double precision.

To official way to use iPython Notebook remotely is to use a [password protected public server](http://ipython.org/ipython-doc/1/interactive/public_server.html), but I prefer using an [ssh tunnel](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh), because it easier to set up, but more hacky.
