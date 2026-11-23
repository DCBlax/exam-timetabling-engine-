.PHONY: install test quick all clean

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

quick:
	python reproduce.py --quick

all:
	python reproduce.py

clean:
	rm -rf results/*.csv results/*.png results/summary.md .pytest_cache
