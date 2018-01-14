all: naive_sa_optimizer twtr_sa_optimizer omp_optimizer seriail_optimizer

naive_sa_optimizer: naive_simulated_annealing.cpp
	g++ -std=c++11 -Wall -O3 naive_simulated_annealing.cpp -o apps/naive_sa_optimizer
	
twtr_sa_optimizer: twtr_simulated_annealing.cpp
	g++ -std=c++11 -Wall -O3 twtr_simulated_annealing.cpp -o apps/twtr_sa_optimizer

omp: optimizer_using_openmp.c
	g++ -O3 -fopenmp -lpthread optimizer_using_openmp.c -o apps/omp_optimizer

serial: optimizer_using_openmp.c 
	g++ -O3 optimizer_using_openmp.c -o apps/serial_optimizer
	
clean:
	rm apps/*
