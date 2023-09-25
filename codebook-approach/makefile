
CXX	= g++
TARGET	= main
OBJS	= main.o UtilityFunctions.o Dataset_multi.o Dataset_single.o
CFLAGS	= -Wall -O2 -std=c++11

.SUFFIXES: .cpp .o

$(TARGET): $(OBJS)
	$(CXX) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cpp
	$(CXX) $(CFLAGS) -c main.cpp

UtilityFunctions.o: UtilityFunctions.cpp
	$(CXX) $(CFLAGS) -c UtilityFunctions.cpp

Dataset_multi.o: Dataset_multi.cpp
	$(CXX) $(CFLAGS) -c Dataset_multi.cpp

Dataset_single.o: Dataset_single.cpp
	$(CXX) $(CFLAGS) -c Dataset_single.cpp

main.o: Dataset_single.h Dataset_multi.h UtilityFunctions.h
UtilityFunction.o: UtilityFunctions.h
Dataset_multi.o: Dataset_multi.h UtilityFunctions.h
Dataset_single.o: Dataset_single.h Dataset_multi.h UtilityFunctions.h

clean:
	rm -f *~ $(TARGET) *.o
	
