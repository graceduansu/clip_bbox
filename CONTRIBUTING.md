# Contributing

Thank you for your interest in contributing to `clip_bbox`! Here are instructions on how to contribute to this project.

## Pre-requisites

* Have [Python 3.8](https://www.python.org/downloads/) installed.
<!-- and ![conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). -->
* Have [Make](https://www.gnu.org/software/make/) installed.
* Have a GitHub account.

Then fork the `clip_bbox` repository on GitHub. Make any desired changes on your own fork first. 

## Development Dependencies

Install this project's Python dependencies by running

    $ make develop

## Makefile
This project is a pure python project using modern tooling. It uses a `Makefile` as a command registry, with the following commands:
- `make`: list available commands
- `make develop`: install and build this library and its dependencies using `pip`
- `make build`: build the library using `setuptools`
- `make check`: check library assets for packaging
- `make lint`: perform static analysis of this library with `flake8` and `black`
- `make format`: autoformat this library using `black`
- `make test`: run automated tests with `pytest`
- `make coverage`: run automated tests with `pytest` and collect coverage information
- `make dist`: package library for distribution
- `make clean`: clean the repository
- `make docs`: clean and build the documentation website

## Before Opening a Pull Request

If you add any features, please add tests for them in `tests/`.

Be sure to check code quality by confirming that the following commands run successfully:

    $ make clean
    $ make lint
    $ make coverage
    $ make check
    $ make docs

## Submitting Changes
Once you are done making changes on your own fork of `clip_bbox`, make a pull request to this repository to submit your changes. 

Be sure to describe your changes in the description of your pull request.
