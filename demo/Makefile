SOURCES=$(shell echo *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
LIBPEDSOURCES=$(shell echo ../libpedsim/*.cpp)
LIBPEDOBJECTS=$(LIBPEDSOURCES:.cpp=.o)
TARGET=demo
QTINCLUDES=-I/usr/include/x86_64-linux-gnu/qt5/QtWidgets -I/usr/include/x86_64-linux-gnu/qt5 -I/usr/include/x86_64-linux-gnu/qt5/QtGui -I/usr/include/x86_64-linux-gnu/qt5/QtCore
INCPATH=-I../libpedsim
CXXFLAGS=-fPIC $(INCPATH) -L../../Libpedsim/src $(QTINCLUDES)
LIBS = -lQt5Widgets -lQt5Gui -lQt5Core -lpedsim
LIBS = -lQt5Widgets -lQt5Gui -lQt5Core
MOCFILES=ParseScenario.moc PedSimulation.moc


all: $(TARGET)

$(TARGET): $(OBJECTS) $(MOCFILES)
	g++ $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LIBPEDOBJECTS) $(LIBS)

%.o: %.cpp %.moc
	g++ $(CXXFLAGS) $(INCPATH) -c -o $@ $<

ParseScenario.moc: ParseScenario.h ParseScenario.cpp
	moc $(DEFINES) $(INCPATH) -i $< -o $@

PedSimulation.moc: PedSimulation.h ParseScenario.cpp
	moc $(DEFINES) $(INCPATH) -i $< -o $@

clean:
	rm $(TARGET) $(OBJECTS) $(MOCFILES)
