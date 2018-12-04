##*********************************####*********************************##
##    ====> ELM Function <====     #### Authors: Lucas Cambuim          ## 
##                                 ####          Fernando Maciano       ##
##                                 ####                                 ##
##*********************************#### E-mail: lfsc@cin.ufpe.br        ##
##                                 ####*********************************##
##                                 #### * Greco - Computer Engineering  ##
##                                 ####   Group                         ##
##                                 #### * CIn - Informatics Center      ##
##                                 #### * UFPE/Brazil                   ##
##*********************************####*********************************##
##*********************************####*********************************##


CXX=g++ -std=c++0x
OPTFLAGS= -O3
CXXFLAGS=-g -Wall -I. -I/usr/local/include $(OPTFLAGS) -I./include -msse3 -I/usr/local/boost_1_52_0
CFLAGS=-Wall $(OPTFLAGS)
LDFLAGS= -L/usr/local/lib $(OPTFLAGS) -L./lib -pthread

LDFLAGS+= `pkg-config opencv --cflags --libs`

SRC = sgbm.o descriptor.o filter.o matrix.o triangle.o elas.o core.o main.o

all: calibration

calibration: $(SRC) $(MODULES)
	$(CXX) $(MODULES) $(SRC) $(LDFLAGS) -pg -ldl -march=native -o main

%.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o calibration

PREFIX ?= /usr

install: all
	install -d $(PREFIX)/bin
	install main  $(PREFIX)/bin

.PHONY: clean all main install
