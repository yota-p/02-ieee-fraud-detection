#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: all clean test jupyter requirements help

## Install Python Dependencies #this may destroy environment!
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete data/interim/, pycache
clean:
	find . -type f -name "*~" -delete
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f data/interim/*

## Delete data/external/, processed/
cleanall: clean
	rm -f data/external/*
	rm -f data/processed/*

## Delete data/*, models/*
eliminate: cleanall
	rm -f models/*
	rm -f data/raw/*

## Create directory
dir:
	mkdir -p data/raw
	mkdir -p data/interim
	mkdir -p data/processed
	mkdir -p data/external

## Run specified version: production
run: dir
	$(PYTHON_INTERPRETER) src/main.py ${ver}

## Run specified version with: nomessage
runs: dir
	$(PYTHON_INTERPRETER) src/main.py ${ver} --nomsg

## Run specified version with: nomessage, debug
runsd: dir
	$(PYTHON_INTERPRETER) src/main.py ${ver} --nomsg --debug

## Start Jupyter-Notebook server
jupyter:
	./startup-jupyter.sh

## Test
test: all
	pytest

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py


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
