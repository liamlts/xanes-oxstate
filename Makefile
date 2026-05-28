ELEMENTS = Ti V Cr Mn Fe Co Ni Cu

.PHONY: data
data:
	@for e in $(ELEMENTS); do \
	  python -m xanes_oxstate.cli --element $$e; \
	done

.PHONY: test
test:
	pytest -q
