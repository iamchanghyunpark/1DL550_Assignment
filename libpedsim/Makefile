TARGET = libpedsim.so
SOURCES = $(shell echo *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
CXXFLAGS = -fPIC -shared -lm

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJECTS)

clean:
	rm $(TARGET) $(OBJECTS)
