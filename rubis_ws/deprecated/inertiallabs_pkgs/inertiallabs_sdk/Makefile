CC      = g++ 
CCFLAGS = -std=c++17
LDFLAGS = -lpthread

default: all

all: debug release

debug: 
	mkdir -p debug
	$(CC) $(CCFLAGS) -Og -ggdb3 -o debug/example example.cpp ILDriver.cpp UDDParser.cpp platforms/linux/SerialPort.cpp platforms/linux/NetClient.cpp $(LDFLAGS)

release: 
	mkdir -p release
	$(CC) $(CCFLAGS) -O3 -g0 -o release/example example.cpp ILDriver.cpp UDDParser.cpp platforms/linux/SerialPort.cpp platforms/linux/NetClient.cpp $(LDFLAGS)

clean:
	rm debug -rf
	rm release -rf
