
# Region Based Hotspot Detection
Source code of paper: (DAC2019 / TCAD2020) Faster Region-based Hotspot Detection.

### Citing 

If you find this repo useful in your research, please consider citing:

    @inproceedings{chen2019faster,
      title={Faster region-based hotspot detection},
      author={Chen, Ran and Zhong, Wei and Yang, Haoyu and Geng, Hao and Zeng, Xuan and Yu, Bei},
      booktitle={2019 56th ACM/IEEE Design Automation Conference (DAC)},
      pages={1--6},
      year={2019},
      organization={IEEE}
    }

## Maintainers
CHEN Ran (chenran1995@link.cuhk.edu.hk)

# Installation

## Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 3.0.0
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow
*   Cython
*   contextlib2
*   cocoapi

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/).
Note that Tensorflow 2.0 is not supported.
I recommend you use tensorflow 1.8.0.
A typical user can install
Tensorflow using one of the following commands:

``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu 16.04 using via apt-get:

``` bash
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib
```

Alternatively, users can install dependencies using pip:

``` bash
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

Note that sometimes "sudo apt-get install protobuf-compiler" will install
Protobuf 3+ versions for you and some users have issues when using 3.5.
If that is your case, you're suggested to download and install Protobuf 3.0.0
(available [here](https://github.com/google/protobuf/releases/tag/v3.0.0)).

## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the tensorflow/models/research/ directory:


``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:


``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file, replacing \`pwd\` with the absolute path of
tensorflow/models/research on your system.

# Training
```bash
bash train.sh GPU_ID
```

# Testing
```bash
bash eval.sh GPU_ID
```

# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
python object_detection/builders/model_builder_test.py
```

## Getting Help


