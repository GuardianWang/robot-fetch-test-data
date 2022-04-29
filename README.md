# Prepare test data from AI2THOR and AI-HABITAT

## Setup

This is developed in Ubuntu 16.04, however Ubuntu 18.04 is recommended because Open3D has a better support for 18.04

```commandline
pip install requirements.py
```

To run CloudRendering in `ai2thor`, 
```commandline
sudo apt-get install libvulkan1
```

If you are using Ubuntu 16.04, visualization may have some bugs, 
you can install a previous version or compile from source ([issue](https://github.com/isl-org/Open3D/issues/1307)).
Or you can also download a pre-built whl from [here](https://gist.github.com/district10/c9d3e2a353b3435a5545b80bf7aba746).