#########
# LINTS #
#########
lint:  ## run static analysis with flake8
	python -m black --check clip_bbox setup.py
	python -m flake8 clip_bbox setup.py

# Alias
lints: lint

format:  ## run autoformatting with black
	python -m black clip_bbox/ setup.py

# alias
fix: format

annotate:  ## run type checking
	python -m mypy ./clip_bbox

#########
# TESTS #
#########
test: ## clean and run unit tests
	python -m pytest -v ./clip_bbox/tests

coverage:  ## clean and run unit tests with coverage
	python -m pytest -v ./clip_bbox/tests --cov=clip_bbox --cov-branch --cov-fail-under=50 --cov-report term-missing

# Alias
tests: test

#########
# CLEAN #
#########
deep-clean: ## clean everything from the repository
	git clean -fdx

clean: ## clean the repository
	rm -rf .coverage coverage cover htmlcov logs build dist *.egg-info .pytest_cache

############################################################################################

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'

.PHONY: lint lints format fix check checks annotate test coverage show-coverage tests show-version deep-clean clean help
