# Build der ball_detector Custom-Parser-.so für DeepStream nvinfer.
# Im DeepStream-Container ausführen (dort sind die Headers vorhanden):
#   make
# Oder von außen: scripts/build_custom_parser.sh (startet Container und baut).

CXX     ?= g++
TARGET  := libnvdsinfer_custom_ball_parser.so
SRC     := nvdsinfer_custom_ball_parser.cpp
OBJ     := $(SRC:.cpp=.o)

# DeepStream-Pfad (im Container typisch /opt/nvidia/deepstream/deepstream)
DS_ROOT ?= /opt/nvidia/deepstream/deepstream
DS_INC  := $(DS_ROOT)/sources/includes
# Optional: TensorRT/CUDA falls der Header sie braucht
CUDA_PATH ?= /usr/local/cuda
TENSORRT_PATH ?= /usr

CXXFLAGS := -std=c++17 -Wall -Wextra -fPIC -O2 -Wno-missing-field-initializers
CXXFLAGS += -I$(DS_INC)
CXXFLAGS += -I$(TENSORRT_PATH)/include
CXXFLAGS += -I$(CUDA_PATH)/include

LDFLAGS  := -shared -Wl,--no-undefined

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)

# Nach models/ kopieren (von Projektroot aus: make install DESTDIR=/app)
install: $(TARGET)
	install -d $(DESTDIR)/models
	install -m 755 $(TARGET) $(DESTDIR)/models/
