#must have python installed!
.PHONY: all clean

.venv:
	python3 -m venv .venv
	pip install -r requirements.txt
clean:
	rm -rf .venv
