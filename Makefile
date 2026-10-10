.DEFAULT_GOAL := help
.PHONY: test system-deps dev-deps deps style lint install help docs lock audit sast sbom api-check security-deps test_vc test_zoo0 test_zoo1 test_zoo2

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := tests TTS notebooks recipes

test_all:	## run tests and don't stop on an error.
	nose2 --with-coverage --coverage TTS tests
	./run_bash_tests.sh

test:	## run tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests

test_vocoder:	## run vocoder tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.vocoder_tests

test_tts:	## run tts tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.tts_tests

test_tts2:	## run tts tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.tts_tests2

test_xtts:
	nose2 -F -v -B --with-coverage --coverage TTS tests.xtts_tests

test_aux:	## run aux tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.aux_tests
	./run_bash_tests.sh

test_zoo:	## run zoo tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.zoo_tests

test_zoo0:	## run zoo tests, shard 0 of 3 (plus voice conversion).
	nose2 -F -v -B --with-coverage --coverage TTS tests.zoo_tests.test_models.test_models_offset_0_step_3
	nose2 -F -v -B --with-coverage --coverage TTS tests.zoo_tests.test_models.test_voice_conversion

test_zoo1:	## run zoo tests, shard 1 of 3.
	nose2 -F -v -B --with-coverage --coverage TTS tests.zoo_tests.test_models.test_models_offset_1_step_3

test_zoo2:	## run zoo tests, shard 2 of 3.
	nose2 -F -v -B --with-coverage --coverage TTS tests.zoo_tests.test_models.test_models_offset_2_step_3

test_vc:	## run voice conversion tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.vc_tests

inference_tests: ## run inference tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.inference_tests

data_tests: ## run data tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.data_tests

test_text: ## run text tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.text_tests

test_failed:  ## only run tests failed the last time.
	nose2 -F -v -B --with-coverage --coverage TTS tests

style:	## update code style.
	black ${target_dirs}
	isort ${target_dirs}

lint:	## run pylint linter.
	pylint ${target_dirs}
	black ${target_dirs} --check
	isort ${target_dirs} --check-only

system-deps:	## install linux system deps
	sudo apt-get install -y libsndfile1-dev

dev-deps:  ## install development deps
	pip install -r requirements.dev.txt

doc-deps:  ## install docs dependencies
	pip install -r docs/requirements.txt

build-docs: ## build the docs
	cd docs && make clean && make build

hub-deps:  ## install deps for torch hub use
	pip install -r requirements.hub.txt

deps:	## install 🐸 requirements.
	pip install -r requirements.txt

security-deps:  ## install the local security scanners.
	pip install -r requirements.security.txt

lock:	## regenerate the per-Python lock files in requirements/ from requirements*.txt.
	for v in 3.10 3.11 3.12 3.13; do \
		uv pip compile --quiet --upgrade --python-version $$v --python-platform linux --no-header --strip-extras --annotation-style line \
			-o requirements/lock-py$$v.txt requirements.txt requirements.dev.txt requirements.ja.txt requirements.notebooks.txt || exit 1; \
	done

audit:	## audit the lock files for known vulnerabilities (CVEs).
	for v in 3.10 3.11 3.12 3.13; do \
		pip-audit --disable-pip --no-deps --progress-spinner off -r requirements/lock-py$$v.txt $(PIP_AUDIT_ARGS) || exit 1; \
	done

sast:	## run static security analysis (CWEs) on the package.
	bandit -c pyproject.toml -r TTS --severity-level medium --confidence-level medium
	ruff check TTS
	semgrep scan --config p/python --config p/flask --config p/trailofbits --severity ERROR --error --metrics off TTS

sbom:	## write a CycloneDX software bill of materials for the installed environment.
	pip-audit --format cyclonedx-json --output sbom.cdx.json --progress-spinner off || true

api-check:	## compare the public API against the dev branch (set AGAINST to override).
	griffe check TTS --against $${AGAINST:-dev} --search . --verbose

install:	## install 🐸 TTS for development.
	pip install -e .[all]

docs:	## build the docs
	$(MAKE) -C docs clean && $(MAKE) -C docs html
