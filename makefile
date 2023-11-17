CC=nvcc
CFLAGS=-lm --std=c++14 -arch sm_61 -I /usr/include/c++/10 -I/home2/cuda/Common
PROGRAM=cudaHisto

SOURCES=main.cu chrono.c
OBJECTS=$(SOURCES:.cu=.o)
SEND_FILES=$(SOURCES) makefile

ARGS_N=100000
ARGS_H=1024
ARGS_NR=1

NV_IP=200.238.130.80
NV_PORT=4450
NV_DIR=$(PROGRAM)_OBJECT_TMP
NV_USERNAME=luccorpp
NV_COMMAND="./$(PROGRAM) $(ARGS_N) $(ARGS_H) $(ARGS_NR)"

KEY_PATH=~/.ssh/nv00_rsa
NV_KEY=$(if $(KEY_PATH),-i $(KEY_PATH), )

DISTDIR=20206150
DISTFILES=$(SOURCES) makefile

# Local targets -----------

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $(PROGRAM) $(CFLAGS)

%.o: %.cu
	$(CC) -c $< $(CFLAGS) -o $@

test: $(PROGRAM)
	./$(PROGRAM) $(ARGS_N) $(ARGS_K)

dist: $(DISTFILES)
	mkdir $(DISTDIR)
	cp $(DISTFILES) $(DISTDIR)
	tar -czvf $(DISTDIR).tgz $(DISTDIR)
	rm -r $(DISTDIR)

clean:
	rm -rf *.o

purge: clean
	rm -rf $(PROGRAM)

# Remote targets -----------

nv-send:
	@test $(NV_USERNAME) || (echo "Missing remote username." ; echo "Run 'export NV_USERNAME=your_username' in order to make remotely." ; exit 1)
	@echo "Creating $(NV_DIR) directory on $(NV_USERNAME)@$(NV_IP):$(NV_PORT)..."
	@ssh $(NV_KEY) $(NV_USERNAME)@$(NV_IP) -p $(NV_PORT) "rm -rf $(NV_DIR); mkdir -p $(NV_DIR)"
ifeq ($(shell echo $$?), 0)
	@echo "> Directory created."
else
	@echo "# Error: failed to create directory."
	exit 1
endif
	@echo "Sending files to $(NV_USERNAME)@$(NV_IP):$(NV_DIR)"
	@scp $(NV_KEY) -P $(NV_PORT) $(SEND_FILES) $(NV_USERNAME)@$(NV_IP):$(NV_DIR)
ifeq ($(shell echo $$?), 0)
	@echo "> Files sent to $(NV_USERNAME)@$(NV_IP):$(NV_DIR)"
else
	@echo "# Error: failed to send files."
	exit 2
endif

nv-all: nv-$(PROGRAM)

nv-$(PROGRAM): nv-send
	@test $(NV_USERNAME) || (echo "Missing remote username." ; echo "Run 'export NV_USERNAME=your_username' in order to make remotely." ; exit 1)
	@echo "Making $(PROGRAM) on $(NV_USERNAME)@$(NV_IP):$(NV_PORT)..."
	@ssh $(NV_KEY) -qt $(NV_USERNAME)@$(NV_IP) -p $(NV_PORT) "cd $(NV_DIR) && make $(PROGRAM) ARGS_N=$(ARGS_N) ARGS_H=$(ARGS_H) ARGS_NR=$(ARGS_NR)"
ifeq ($(shell echo $$?), 0)
	@echo "> Make done. Run 'make nv-run' to execute."
else
	@echo "# Error: failed to make."
	exit 3
endif

nv-test: nv-send
	@test $(NV_USERNAME) || (echo "Missing remote username." ; echo "Run 'export NV_USERNAME=your_username' in order to make remotely." ; exit 1)
	@ssh $(NV_KEY) -qt $(NV_USERNAME)@$(NV_IP) -p $(NV_PORT) "cd $(NV_DIR) && make test ARGS_K=$(ARGS_K) ARGS_N=$(ARGS_N)"

nv-run:
	@test $(NV_USERNAME) || (echo "Missing remote username." ; echo "Run 'export NV_USERNAME=your_username' in order to make remotely." ; exit 1)
	@echo "Running $(PROGRAM) on $(NV_USERNAME)@$(NV_IP):$(NV_PORT)..."
	@ssh $(NV_KEY) -qt $(NV_USERNAME)@$(NV_IP) -p $(NV_PORT) "cd $(NV_DIR) && $(NV_COMMAND)"

nv-clean:
	@test $(NV_USERNAME) || (echo "Missing remote username." ; echo "Run 'export NV_USERNAME=your_username' in order to make remotely." ; exit 1)
	@ssh $(NV_KEY) $(NV_USERNAME)@$(NV_IP) -p $(NV_PORT) "rm -rf $(NV_DIR)"

# --------------------------

.PHONY: all test clean purge nv-send nv-all nv-$(PROGRAM) nv-test nv-run nv-clean
