# Compiler: CC is the Cray C++ wrapper
CXX      = CC
CXXFLAGS = -O2 -fopenmp -std=c++17 -Wall

.PHONY: all clean

# Build all three binaries
all: seq data task

# Sequential
seq: seq.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Data parallel (OpenMP)
data: data.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Task parallel (OpenMP)
task: task.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f seq data task