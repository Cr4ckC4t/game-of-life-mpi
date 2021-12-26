# Game of Life - using MPI

A distributed-computing version of the classic "Game of Life" using the Message Passing Interface (MPI) on Linux. The code includes a stub for leveraging a Pi Cluster to draw the game on 8x8 Led Pi HATs.

The code is commented and includes additional information on how to use, compile and modify the program. The program features no command line parameters but can be configured to some extend via the `#define`s in the beginning.

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

## Example
![Game of Life demonstration](game-of-life-demonstration.gif)
