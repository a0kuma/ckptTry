TEX := hello
BUILD_DIR := build
GEN_DIR := $(BUILD_DIR)/generated
WORKSPACE_CONDA_ENV_NAME = sum26
CPP_SRCS := $(shell sed -n 's/.*\\[Vv]erbatim[Ii]nput\(\[[^]]*\]\)\?{\([^}]*\.cpp\)}.*/\2/p' $(TEX).tex)
PY_SRCS := $(shell sed -n 's/.*\\[Vv]erbatim[Ii]nput\(\[[^]]*\]\)\?{\([^}]*\.py\)}.*/\2/p' $(TEX).tex)
CPP_STDOUT_TXTS := $(foreach src,$(CPP_SRCS),$(GEN_DIR)/$(basename $(notdir $(src)))-stdout.txt)
PY_STDOUT_TXTS := $(foreach src,$(PY_SRCS),$(GEN_DIR)/$(basename $(notdir $(src)))-stdout.txt)
STDOUT_TXTS := $(CPP_STDOUT_TXTS) $(PY_STDOUT_TXTS)

CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -pedantic
CONDA ?= conda
PYTHON ?= python
XELATEX ?= xelatex
XELATEX_FLAGS ?= -interaction=nonstopmode -halt-on-error

.PHONY: all pdf run clean distclean

all: pdf

pdf: $(TEX).pdf

$(TEX).pdf: $(TEX).tex heart-border.tex $(STDOUT_TXTS) fonts/ChenYuluoyan-2.0-Thin.ttf
	$(XELATEX) $(XELATEX_FLAGS) $(TEX).tex

define CPP_RULES
$(BUILD_DIR)/$(basename $(notdir $(1))): $(1) | $(BUILD_DIR)
	$$(CXX) $$(CXXFLAGS) $$< -o $$@

$(GEN_DIR)/$(basename $(notdir $(1)))-stdout.txt: $(BUILD_DIR)/$(basename $(notdir $(1))) | $(GEN_DIR)
	$$< > $$@
endef

$(foreach src,$(CPP_SRCS),$(eval $(call CPP_RULES,$(src))))

define PY_RULES
$(GEN_DIR)/$(basename $(notdir $(1)))-stdout.txt: $(1) | $(GEN_DIR)
	$$(CONDA) run -n $$(WORKSPACE_CONDA_ENV_NAME) $$(PYTHON) $$< > $$@.tmp 2>&1
	perl -pe 's/\x1b\[[0-9;]*m//g' $$@.tmp > $$@
	rm -f $$@.tmp
endef

$(foreach src,$(PY_SRCS),$(eval $(call PY_RULES,$(src))))

$(BUILD_DIR) $(GEN_DIR):
	mkdir -p $@

run: $(STDOUT_TXTS)
	@if [ -n "$(STDOUT_TXTS)" ]; then \
		for output in $(STDOUT_TXTS); do \
			printf '%s\n' "$$output"; \
			cat "$$output"; \
		done; \
	fi

clean:
	rm -rf $(BUILD_DIR) $(TEX).aux $(TEX).log $(TEX).out

distclean: clean
	rm -f $(TEX).pdf
