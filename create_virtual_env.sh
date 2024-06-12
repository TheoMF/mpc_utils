#!/bin/bash -eux

if [[ ! -d venv ]]
then
    python3 -m venv venv
fi

source venv/bin/activate

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

