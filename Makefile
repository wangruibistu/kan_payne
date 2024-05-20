clean:
	rm -rf build kan_payne.egg-info
	rm -rf .coverage .pytest_cache

install: clean
	pip install .
	rm -rf build kan_payne.egg-info

uninstall:
	pip uninstall kan_payne -y

test:
	coverage run -m  pytest . --import-mode=importlib --cov-report=html --cov-report=term-missing
	coverage report -m
	rm -rf .coverage .pytest_cache
