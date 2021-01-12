TARGET = libpedsim.so
SOURCES = $(shell echo *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
CUDA_SOURCES = $(shell echo *.cu)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.co)
CXXFLAGS = -fPIC -shared -lm -fopenmp
CUDA_NVCC_FLAGS = --compiler-options -fPIC,-shared -fopenmp

all: $(TARGET)

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJECTS) $(CUDA_OBJECTS)

%.co: %.cu
	echo $@ $<
	nvcc $(CUDA_NVCC_FLAGS) $(DEBUGFLAGS) -c -o $@ $<


clean:
	rm $(TARGET) $(OBJECTS)
