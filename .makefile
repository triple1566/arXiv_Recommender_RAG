#must have python installed!
.PHONY: all venv clean

all: venv requirements.txt
    pip install -r requirements.txt

venv:
    python3 -m venv venv
    source ./venv/bin/activate

clean:
    rm -rf venv