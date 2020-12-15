.PHONY: clean libpedsim demo

all: libpedsim demo

libpedsim:
	make -C libpedsim

demo:
	make -C demo

clean:
	make -C libpedsim clean
	make -C demo clean
