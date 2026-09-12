# Build flags for the cgo dependencies.
#
# whisper.cpp is statically linked, so every ggml archive has to be named
# explicitly -- upstream's own Makefile omits -lggml-cpu, -lggml-base and
# -framework Accelerate, which link fine on a shared build and fail on a static
# one. Metal shaders are embedded in the library (GGML_METAL_EMBED_LIBRARY=ON)
# so no resource path is needed at runtime.
#
# onnxruntime needs no flags at all: the Go binding dlopens the dylib at
# runtime, located via ONNXRUNTIME_LIB.

REPO      := $(abspath $(CURDIR)/..)
WHISPER   := $(REPO)/third_party/whisper.cpp
BUILD     := $(WHISPER)/build_go

export C_INCLUDE_PATH := $(WHISPER)/include:$(WHISPER)/ggml/include
export LIBRARY_PATH   := $(BUILD)/src:$(BUILD)/ggml/src:$(BUILD)/ggml/src/ggml-blas:$(BUILD)/ggml/src/ggml-metal

LDFLAGS := -ldflags "-extldflags '-lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas \
	-framework Foundation -framework Metal -framework MetalKit -framework Accelerate'"

.PHONY: build run test tidy whisper clean

build:
	go build $(LDFLAGS) -o bin/glydi ./cmd/glydi

run: build
	./bin/glydi

# The identity store and LLM client have no cgo dependencies, so they test
# without any of the above.
test:
	go test ./internal/identity/... ./internal/llm/...

test-all:
	go test $(LDFLAGS) ./...

tidy:
	go mod tidy

# Rebuild whisper.cpp from source. Only needed on a fresh clone -- the built
# archives are committed under third_party.
whisper:
	cmake -S $(WHISPER) -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
		-DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF \
		-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
	cmake --build $(BUILD) --target whisper -j

clean:
	rm -rf bin
