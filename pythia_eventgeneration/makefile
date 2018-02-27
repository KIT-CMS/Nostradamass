CC=g++
CFLAGS=-c -Wall -I/cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc62-opt/include -g -c -Wall -I$(ROOTSYS)/include -Wno-delete-non-virtual-dtor
LDFLAGS=-L/cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc62-opt/lib -lpythia8 -lhdf5 -lhdf5_cpp -L$(ROOTSYS)/lib -lTree
SOURCES=eventgeneration.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=eventgeneration

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cc.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(EXECUTABLE)

