#!/bin/zsh

export HOME_DIR=/Users/chuyen/projects/model-validation-app/capp

CAPP_ENV_DIR=$HOME_DIR/capp-venv

export PYTHONPATH=$HOME_DIR

export JUPYTER_CONFIG_DIR=$CAPP_ENV_DIR/share/jupyter/config
export JUPYTER_DATA_DIR=$CAPP_ENV_DIR/share/jupyter/data
export JUPYTER_RUNTIME_DIR=$CAPP_ENV_DIR/share/jupyter/runtime

source $CAPP_ENV_DIR/bin/activate

jupyter notebook --no-browser
