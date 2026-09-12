# Standalone build for the Swift TTS helper.
#
#   make            # build ./ttsd
#   make clean
#
# The top-level Makefile can drive this with:
#   ttsd:
#           $(MAKE) -C cmd/ttsd

SWIFTC ?= swiftc
SWIFT_FLAGS ?= -O

.PHONY: all clean

all: ttsd

ttsd: ttsd.swift
	$(SWIFTC) $(SWIFT_FLAGS) $< -o $@

clean:
	rm -f ttsd
