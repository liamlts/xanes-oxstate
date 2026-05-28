ELEMENTS = Ti V Cr Mn Fe Co Ni Cu

.PHONY: data
data:
	@for e in $(ELEMENTS); do \
	  python -m xanes_oxstate.cli build-data --element $$e; \
	done

.PHONY: train
train:
	@for e in $(ELEMENTS); do \
	  python -m xanes_oxstate.cli evaluate --element $$e --config configs/$$e.yaml; \
	done

.PHONY: figures
figures:
	python scripts/make_figures.py

.PHONY: all
all: data train figures

.PHONY: test
test:
	pytest -q
