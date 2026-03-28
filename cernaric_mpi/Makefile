# Compiler: CC is the Cray C++ wrapper, auto-links MPI libraries
CXX      = CC

# -O2: optimization level 2 (fast code, reasonable compile time)
# -fopenmp: enable OpenMP pragma support
# -std=c++17: enable C++17 (needed for structured bindings like auto [r,c])
# -Wall: enable all common compiler warnings
CXXFLAGS = -O2 -fopenmp -std=c++17 -Wall

# Name of the output binary
TARGET   = cernaric_mpi

# Source file to compile
SRC      = cernaric_mpi.cpp

# .PHONY: tells Make these aren't real files, so it always runs the recipe
# (without this, a file called 'clean' would block 'make clean')
.PHONY: all clean pack

# Default rule: typing 'make' builds the target binary
all: $(TARGET)

# Build rule: recompile if source is newer than binary
# $@ = the target (cernaric_mpi), $< = the first dependency (cernaric_mpi.cpp)
# expands to: CC -O2 -fopenmp -std=c++17 -Wall -o cernaric_mpi cernaric_mpi.cpp
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Delete the binary (-f = don't error if file doesn't exist)
clean:
	rm -f $(TARGET)

# Create the submission archive for ProgTest
# tar c=create z=gzip-compress f=output-filename
pack:
	tar czf cernaric-PDP-26-4.tar.gz $(SRC) Makefile mpi_job.sh