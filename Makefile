.PHONY: setup test physics demo clean

setup:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install -e ".[dev]"

test:
	python -m pytest tests/ -q

physics:
	python -m cbmx.cli physics

# Commission, learn normal, then plant each fault in turn.
demo:
	python -m cbmx.cli demo

# Real data. fetch needs network; eval needs the fetch to have run.
cwru:
	python tools/fetch_cwru.py --out data/cwru

eval:
	python tools/eval_cwru.py --data data/cwru

clean:
	rm -rf __pycache__ .pytest_cache src/cbmx/__pycache__
