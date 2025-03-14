#must have python installed!
SHELL := /bin/bash
.PHONY: clean
PIP= .venv/bin/pip

.venv: requirements.txt
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

clean:
	rm -rf .venv
