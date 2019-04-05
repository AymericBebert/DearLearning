# DearLearning

Project idea: use Deep Learning to ear the difference between instruments

## Setup

### Requirements

Developed with Python 3.7 and should be compatible with Python 3.6+

Developed on macOS and Ubuntu. Not sure about compatibility with other systems

Install the requirements with the usual:

```sh
pip3 install -r requirements.txt
```

### Edit configuration

To adapt the configuration file to your environment, you need to create a `local.py` file
in the `config` directory, next to the `default.py` config file.

You can overwrite every item of the default config, maybe paths, log config,...

## Run

You should be able to run the code with:

```sh
./src/main.py
```

You may have to run `export PYTHONPATH=$(pwd)` beforehand for the imports to work.

You can also run the realtime recognition with:

```sh
./src/realtime.py
```

At the end of the project we aldo pushed the notebook with tests, MLP and CNN, named `Tests.ipynb`

You can view its content easily with `Tests.html`, or open it in jupyter notebook.
