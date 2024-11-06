test:
	poetry run pytest --cov --cov-config=.coveragerc --cov-report term --cov-report html

update_test:
	UPDATE_FILE_FIXTURES=TRUE poetry run pytest --cov --cov-config=.coveragerc --cov-report term --cov-report html

install:
	poetry install

lint:
	poetry run pylint --disable=R,C .

check:
	poetry run mypy .

format:
	poetry run ruff format .

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ | xargs rm -rf

all: clean install lint format test
