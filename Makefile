#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: all clean test jupyter data lint requirements help

## Install Python Dependencies #this may destroy environment!
#requirements: test_environment
#	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete cache files
clean:
	find . -type f -name "*~" -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete processed & interim files
distclean: clean
	rm -f data/raw/*.pickle
	rm -f data/interim/*

## Delete final outputs
eliminate: distclean
	rm -f data/raw/*
	rm -f data/processed/*
	rm -f models/*.model

## Start Jupyter-Notebook
jupyter:
	nohup jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root >> notebooks/jupyter.log 2>&1 &
	echo 'gcloud compute ssh HOST -- -N -L 8888:localhost:8888'
	sleep 3s
	tail -n 2 notebooks/jupyter.log


#data: requirements
#	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Preprocessing
data/processed/processed.pickle: $(DATA_RAW)
	$(PYTHON_INTERPRETER) src/data/preprocess.py $< $@

## Export report
#reports/figures/exploratory.png: data/processed/processed.pickle
#	$(PYTHON_INTERPRETER) src/visualization/exploratory.py $< $@

## Train model
#models/random_forest.model: data/processed/processed.pickle
#	$(PYTHON_INTERPRETER) src/models/train_model.py $< $@

## All
#all: data/raw/iris.csv data/processed/processed.pickle reports/figures/exploratory.png models/random_forest.model

## Test
test: all
	pytest

## Lint using flake8
#lint:
#	flake8 src

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
