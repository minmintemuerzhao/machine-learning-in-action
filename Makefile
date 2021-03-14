.PHONY: clean deps dev flake8 pylint lint test pre-commit dist upload

PIPRUN := $(shell command -v pipenv > /dev/null && echo pipenv run)
TARGET := tests customer_insights *.py

clean:
	find . -name '*.pyc' -print0 | xargs -0 rm -f
	find . -name '*.swp' -print0 | xargs -0 rm -f
	find . -name '__pycache__' -print0 | xargs -0 rm -rf
	find . -name '.pytest_cache' -print0 | xargs -0 rm -rf

deps:
	${PIPRUN} pip install -r requirements.txt

dev:
	${PIPRUN} pip install -r requirements-dev.txt
	${PIPRUN} pre-commit install

flake8:
	${PIPRUN} flake8 \
		--statistics \
		--inline-quotes 'double' \
		${TARGET}

pylint:
	${PIPRUN} pylint ${TARGET}

lint: flake8 pylint

test:
	${PIPRUN} pytest --cov=customer_insights

pre-commit:
	${PIPRUN} pre-commit run --all-files

dist:
	${PIPRUN} python setup.py sdist bdist_wheel
