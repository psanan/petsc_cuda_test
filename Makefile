EXNAME = runme

OBJECTS = \
			main.o \
			set_array.o \

all : $(EXNAME)

ifndef PETSC_DIR
	$(error PETSC_DIR must be set)
endif
include ${PETSC_DIR}/share/petsc/Makefile.user

$(EXNAME) : $(OBJECTS)
	$(LINK.cc) -o $@ $^ $(LDLIBS) $(CUDA_LIB)

clean :
	rm -f $(EXNAME)
	rm -f $(OBJECTS)

.PHONY: all clean
