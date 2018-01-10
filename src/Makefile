all: santa twtr omp serial

santa: santa.cpp
	g++ -std=c++11 -Wall -O3 santa.cpp -o apps/app
	
twtr: twtr.cpp
	g++ -std=c++11 -Wall -O3 twtr.cpp -o apps/twtr

omp: improver.c
	g++ -O3 -fopenmp -lpthread improver.c -o apps/omp

serial: improver.c
	g++ -O3 improver.c -o apps/serial
	
clean:
	rm apps/*
