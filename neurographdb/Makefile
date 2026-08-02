CXX      = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -fPIC
PY       ?= python3.13
PY_INCLUDES   := $(shell $(PY) -m pybind11 --includes)
PY_EXT_SUFFIX := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_MODULE = ngdb/_ngdb_core$(PY_EXT_SUFFIX)

SRC   = src
BUILD = build
CORE  = $(SRC)/bm25.cpp $(SRC)/dense.cpp $(SRC)/graph.cpp
OBJS  = $(patsubst $(SRC)/%.cpp,$(BUILD)/%.o,$(CORE))

.PHONY: all clean
all: $(PY_MODULE)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC)/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(SRC) -c -o $@ $<

$(PY_MODULE): $(OBJS) $(SRC)/bindings.cpp
	$(CXX) $(CXXFLAGS) -shared -undefined dynamic_lookup \
	  $(PY_INCLUDES) -I$(SRC) -o $@ $(OBJS) $(SRC)/bindings.cpp
	@echo "  빌드 완료 → $(PY_MODULE)"

clean:
	rm -rf $(BUILD) $(PY_MODULE)
