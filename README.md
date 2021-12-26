# Game of Life - using MPI

A distributed-computing version of the classic "Game of Life" using the Message Passing Interface (MPI) on Linux. The code includes a stub for leveraging a Pi Cluster to draw the game on 8x8 Led Pi HATs.

The code is commented and includes additional information on how to use, compile and modify the program. The program features no command line parameters but can be configured to some extend via the `#define`s in the beginning.

## Why?

The concept of distributing tasks on multiple processors to maximize throughput is here simulated using [John Conway's Game of Life](https://playgameoflife.com/info) to practice the use of the MPI library in C.

## Usage

The program distributes the computation of different regions in the Game of Life on the specified amount of processors.

In order to compile and run the code, you'll need the `mpi-default-dev` package (standard in debian repositories).
```bash
sudo apt install mpi-default-dev
```
**Compile the source code:**
```bash
mpicc -o gol-mpi -Wall -lm gol-mpi.c
```
**Execute:**
> Note that in order to execute the following command you will require at least 4 available cores (can be checked with `lscpu`).
```bash
mpirun -np 4 ./gol-mpi  # -np specifies the amount of processors to use
```

## Examples

Running GoL on a 27x27 grid with 9 processors using a glider as start formation and coloring each processors region differently:
![Game of Life demonstration](game-of-life-demonstration.gif)

Running GoL on a 32x32 grid with 4 processors using a random start formation (35th iteration):
![Game of Life demonstration](game-of-life-demonstration.png)
