CC      = clang

RUBY_HDR      := $(shell ruby -e 'print RbConfig::CONFIG["rubyhdrdir"]')
RUBY_ARCH_HDR := $(shell ruby -e 'print RbConfig::CONFIG["rubyarchhdrdir"]')
CPPFLAGS      := -I$(RUBY_HDR) -I$(RUBY_ARCH_HDR)

SRC := ext/main.c
OBJ := build/main.o

.PHONY: all clean

all: $(OBJ)

$(OBJ): $(SRC)
	@mkdir -p build
	$(CC) $(CPPFLAGS) -c $(SRC) -o $(OBJ)

compile_commands.json: clean
	bear -- $(MAKE) all

clean:
	rm -rf build compile_commands.json
