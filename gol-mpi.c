/*
*       Game of Life Implementation using MPI
*
*       Usage Example:
*
*       Compile with: mpicc -o gol-mpi gol-mpi.c -lm -Wall
*       Execute with: mpirun -np 4 ./gol-mpi
*
*       This will distribute a 32x32 grid on 4 cores and print 500 iterations of GoL on the console.
*
*
*       ----------------------------------------------------------------
*       Assumptions:
*               The playing field is a square of size NxN with N being at least 8.
*               The number of processors must be chosen so that each processor
*               can process an equal-sized sub-square of the grid.
*
*       N (the width of the playing field) can be set with the define "GRID_WIDTH" (default: 32).
*       M (the number of processors) can be set with mpirun on the CLI with "-np <number of processors>".
*
*       Note: You will be warned when using an invalid combination of N and M
*
*       Example: N=9  M=9  -> 9 processors calculate 3x3 squares
*                N=32 M=4  -> 4 processors calculate 16x16 squares
*                N=11 n/a  -> not valid
*
*       Basic playing grid:
*                N=32 M=16
*                _______________32________________
*               /                                 \_
*               xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
*               ... 0    ... 1    ... 2    ... 3 <------- Processor rank
*               xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
*                                                   |
*             +-xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
*             | xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
* Pi LED HAT -| ... 4    ... 5    ... 6    ... 7    |
*    8x8      | xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
*             +-xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
*                                                    > 32
*               xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
*               ... 8    ... 9    ...10    ...11    |
*               xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
*                                                   |
*               xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx |
*               ...12    ...13    ...14    ...15    |
*               xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx_|
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <time.h> // Seed rand() with time(NULL)

/* User controllable parameters */
#define GRID_WIDTH 32                           // Set the width of the square grid (N)
                                                // This value should not be a prime number and at least 8.

#define TOTAL_GRID_SIZE GRID_WIDTH*GRID_WIDTH   // DO NOT CHANGE - calculate the size of the grid

#define N_GENERATIONS 500                       // Set the amount of generations to simulate
#define GEN_DELAY_MS  100                       // Set the delay per generation in milliseconds

#define START_RANDOM 1                          // Set to 1 to fill the grid randomly at start. If
                                                // set to 0 then a glider will be spawned.

#define DISTRIBUTE_DRAW 0                       // Set to 1 to distribute the drawing of the grid over
                                                // the Raspberry Pis (ignores COLOR_SUB_GRIDS). Set
                                                // to 0 to let the root processor gather and draw the
                                                // entire grid (considers COLOR_SUB_GRIDS).

#define COLOR_SUB_GRIDS 1                       // Set to 1 to activate a colored grid. Set to 0 to
                                                // get the default BLACK/WHITE output.

const char *ARR_COLORS[] = {                    // Background colors to use for coloring sub grids
        "\033[48;5;1m",         // RED          // (when activated)
        "\033[48;5;2m",         // GREEN
        "\033[48;5;3m",         // YELLOW
        "\033[48;5;4m",         // BLUE
        "\033[48;5;5m",         // PINK
        "\033[48;5;6m",         // TURQ
        "\033[48;5;9m",         // ORANGE
        "\033[48;5;87m",        // CYAN
        "\033[48;5;218m"        // ROSE
};

/* System parameters - DO NOT CHANGE  */
#define NUM_COLORS (sizeof(ARR_COLORS) / sizeof(const char *)) // Get the size of ARR_COLORS

#define S_TOPLEFT       "\033[H"        // Set cursor to top left
#define C_RST           "\033[0;39m"    // Reset color code to default
#define C_B_BLACK       "\033[0;40m"    // Set background color black
#define C_B_WHITE       "\033[0;47m"    // Set background color white

#define TAG_UL  10 // Receiving value for upper left corner
#define TAG_UR  20 // Receiving value for upper right corner
#define TAG_DL  30 // Receiving value for down left corner
#define TAG_DR  40 // Receiving value for down right corner
#define TAG_UP  50 // Receiving values for upper border
#define TAG_DO  60 // Receiving values for downside border
#define TAG_RI  70 // Receiving values for right border
#define TAG_LE  80 // Receiving values for left border


/**
 * @brief Transform one grid into a concatenation of several smaller squares.
 *
 * @param grid          A pointer to the entire grid in normal format.
 * @param edge_length   The length of one side of a square.
 */
void transform_for_distribution(int grid[TOTAL_GRID_SIZE], int edge_length);

/**
 * @brief Tansform many concatenated smaller grids back into one grid.
 *
 * @param grid          A pointer to the grid that was gathered from all processors.
 * @param edge_length   The length of one side of a square.
 */
void transform_from_distribution(int grid[TOTAL_GRID_SIZE], int edge_length);

/**
 * @brief Draw the entire grid.
 *
 * This function can be used to draw the entire grid that was scattered, computed and
 * then gathered by the root processor. To let each processor draw its own grid use
 * distributed_draw_grid instead.
 *
 *
 * @param grid          A pointer to the grid in normal format.
 * @param edge_length   The length of the subgrids. This will be used to color the
 *                      distributed blocks if COLOR_SUB_GRIDS is activated.
 */
void draw_grid(int grid[TOTAL_GRID_SIZE], int edge_length);

/**
 * @brief Distributed version of draw_grid. Draw the local grid.
 *
 * This function is used to draw the local grid on a Raspberry Pi LED HAT.
 * The local grid should therefore be 8x8. (For use on a Pi Cluster only.)
 *
 * @param local_grid    A poiner to the local (8x8) grid.
 * @param edge_length   Length of one side of the grid. This should always be
 *                      8 as the Raspberry Pi HAT is an 8x8 LED matrix. Otherwise
 *                      the drawing will be supressed.
 */
void draw_local_grid(int *local_grid, int edge_length);

/**
 * @brief Get the eight neighbours of a processor (with wrap around).
 *
 * Update proc_ids with the 8 neighbour-ids that surround "rank" in a square of a total of n_proc processors.
 * The neighbour positions are denoted like this:
 *
 *      0   1   2
 *      3 rank  4  - the eight neighbours (0 being the upper left adjacent processor)
 *      5   6   7
 *
 * @param proc_ids      An array that will be filled with the neighbour ranks for each direction.
 * @param rank          The rank to find the neighbours for.
 * @param n_procs       The total amount of processors. Must be the square of an integer.
 */
void get_neighbour_ids(int proc_ids[8], int rank, int n_procs);

/**
 * @brief Update each cell of a local grid, taking into account surrounding grids.
 *
 * @param g             A pointer to the local grid.
 * @param width         The length of one side of the local grid.
 * @param ul            The corner value of the top left processor.
 * @param ur            The corner value of the top right processor.
 * @param dl            The corner value of the lower left processor.
 * @param dr            The corner value of the lower right processor.
 * @param ups           Array with adjacent values of the processor above.
 * @param downs         Array with adjacent values of the processor below.
 * @param lefts         Array with adjacent values of the processor to the left.
 * @param rights        Array with adjacent values of the processor to the right.
 */
void update_local_grid(int *g, int width, int ul, int ur, int dl, int dr, int *ups, int *downs, int *lefts, int *rights);

/**
 * @brief Main entry point.
 *
 * @param argc          Argument counter.
 * @param argv          Argument vector.
 */
int main(int argc, char** argv) {

        /* MPI Initialisation */
        int size, my_rank;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total amount of processors
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Get the current rank
        if (ceilf(sqrt(size)) != sqrt(size)) {
                fprintf(stdout, "M is not square, aborting (processors = %d).\n",size);
                exit(1);
        }

        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len); // Get the processor name

        /* Determine local responsibility */
        const int local_grid_size = TOTAL_GRID_SIZE / size; // Get local grid size (distribute grid on M processors)
        if (ceilf(sqrt(local_grid_size)) != sqrt(local_grid_size)) {
                fprintf(stdout, "Local grid is not a square (local_grid_edge = %f).\n", sqrt(local_grid_size));
                exit(1);
        }
        const int local_edge_length = (int)sqrt(local_grid_size); // Get the length of one side of the local grid

        /* Allocate memory for the local grid */
        int *local_grid = malloc(sizeof(int) * local_grid_size);

        /* Initialise entire grid and communicate it to all processors */
        int grid[TOTAL_GRID_SIZE] = {0};
        if (!my_rank) {
                /* Proc 0 initialises and distributes data */
                if (!START_RANDOM && GRID_WIDTH > 3) {
                        /* Create a glider in the upper left corner */
                        grid[GRID_WIDTH+3]=1;
                        grid[GRID_WIDTH*2+1]=1;
                        grid[GRID_WIDTH*2+3]=1;
                        grid[GRID_WIDTH*3+2]=1;
                        grid[GRID_WIDTH*3+3]=1;
                } else {
                        srand(time(NULL)); // Seed PRNG
                        for (int i=0; i<TOTAL_GRID_SIZE; i++)
                                grid[i] = rand()%2; // Set random 0 or 1
                }

                /* Transform grid for easy distribution */
                transform_for_distribution(grid, local_edge_length);
        }

        /* Distribute the entire grid across all processors */
        MPI_Scatter(grid, local_grid_size, MPI_INT, local_grid, local_grid_size, MPI_INT, 0, MPI_COMM_WORLD);
        /* Each processor does now have a part of the grid in local_grid */

        fprintf(stdout, "[%d|%d] (%s): Local grid size = %dx%d\n", my_rank, size, processor_name, local_edge_length, local_edge_length);

        /* Synchronize all processors */
        MPI_Barrier(MPI_COMM_WORLD);
        if (!my_rank) {
                fprintf(stdout, "\nReady to start? Press ENTER to continue.");
                fflush(stdout);
                getchar();
                system("clear");
        }

        /* Determine all eight neighbour processors */
        int neigh_procs[8] = {0};
        get_neighbour_ids(neigh_procs, my_rank, size);

        /* Prepare some values for the context from the other processes */
        int up_left, up_right, down_left, down_right;
        int *my_ups = malloc(sizeof(int) * local_edge_length);
        int *my_lefts = malloc(sizeof(int) * local_edge_length);
        int *my_rights = malloc(sizeof(int) * local_edge_length);
        int *my_downs = malloc(sizeof(int) * local_edge_length);

        /* Game of Life - Loop */
        for (int gen=0; gen < N_GENERATIONS; gen++) {
                /* Synchronize all processors */
                MPI_Barrier(MPI_COMM_WORLD);

                /* Draw the grid */
                if (DISTRIBUTE_DRAW)
                        /* Let each processor draw parts of the grid (for use on a raspberry pi cluster) */
                        draw_local_grid(local_grid, local_edge_length);
                else {
                        /* Gather all distributed fields so proc 0 can display everything */
                        MPI_Gather(local_grid, local_grid_size, MPI_INT, grid, local_grid_size, MPI_INT, 0, MPI_COMM_WORLD);

                        /* Processor 0 */
                        if (!my_rank) {
                                /* Transform distributed grid back into one grid */
                                transform_from_distribution(grid, local_edge_length);
                                /* Display grid */
                                draw_grid(grid,local_edge_length);
                                /* Print generation */
                                fprintf(stdout, "Generation: %d|%d\n", gen, N_GENERATIONS-1);
                        }
                }

                /* Provide and collect all required contexts for/from the other processors */

                /* Expose own corners */
                MPI_Send(&local_grid[0], 1, MPI_INT, neigh_procs[0], TAG_DR, MPI_COMM_WORLD); // expose up-left
                MPI_Send(&local_grid[local_edge_length-1], 1, MPI_INT, neigh_procs[2], TAG_DL, MPI_COMM_WORLD); // expose up-right
                MPI_Send(&local_grid[local_grid_size-local_edge_length], 1, MPI_INT, neigh_procs[5], TAG_UR, MPI_COMM_WORLD); // expose lower left
                MPI_Send(&local_grid[local_grid_size-1], 1, MPI_INT, neigh_procs[7], TAG_UL, MPI_COMM_WORLD); // expose lower right

                /* Get own borders */
                for (int i=0; i<local_edge_length;i++) {
                        my_ups[i] = local_grid[i];
                        my_lefts[i] = local_grid[i*local_edge_length];
                        my_rights[i] = local_grid[(i+1)*local_edge_length-1];
                        my_downs[i] = local_grid[local_grid_size-local_edge_length+i];
                }

                /* Expose own borders */
                MPI_Send(my_ups, local_edge_length, MPI_INT, neigh_procs[1], TAG_DO, MPI_COMM_WORLD); // expose ups
                MPI_Send(my_lefts, local_edge_length, MPI_INT, neigh_procs[3], TAG_RI, MPI_COMM_WORLD); // expose lefts
                MPI_Send(my_rights, local_edge_length, MPI_INT, neigh_procs[4], TAG_LE, MPI_COMM_WORLD); // expose rights
                MPI_Send(my_downs, local_edge_length, MPI_INT, neigh_procs[6], TAG_UP, MPI_COMM_WORLD); // expose downs

                /* Collect adjacent corners */
                MPI_Recv(&up_left, 1, MPI_INT, neigh_procs[0], TAG_UL, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive upper left
                MPI_Recv(&up_right, 1, MPI_INT, neigh_procs[2], TAG_UR, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive upper right
                MPI_Recv(&down_left, 1, MPI_INT, neigh_procs[5], TAG_DL, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive down left
                MPI_Recv(&down_right, 1, MPI_INT, neigh_procs[7], TAG_DR, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive down right

                /* Collect adjacent borders */
                MPI_Recv(my_ups, local_edge_length, MPI_INT, neigh_procs[1], TAG_UP, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive ups
                MPI_Recv(my_downs, local_edge_length, MPI_INT, neigh_procs[6], TAG_DO, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive downs
                MPI_Recv(my_lefts, local_edge_length, MPI_INT, neigh_procs[3], TAG_LE, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive lefts
                MPI_Recv(my_rights, local_edge_length, MPI_INT, neigh_procs[4], TAG_RI, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive rights

                /* Update local grid */
                update_local_grid(local_grid, local_edge_length, up_left, up_right, down_left, down_right, my_ups, my_downs, my_lefts, my_rights);

                /* Generation delay */
                usleep(GEN_DELAY_MS*1000);
        }

        /* Free the pointers */
        free(local_grid);
        free(my_ups);
        free(my_lefts);
        free(my_rights);
        free(my_downs);

        /* MPI Finalisation */
        MPI_Finalize();

        return 0;
}

void update_local_grid(int *g, int width, int ul, int ur, int dl, int dr, int *ups, int *downs, int *lefts, int *rights) {

        /* Prepare a copy of the local grid with an additional border for the context values */
        int size = (width+2)*(width+2);
        int *cg = malloc(sizeof(int) * size);

        /* Fill the copy with local_grid and border values */
        for (int y=0; y<(width+2); y++) {
                for (int x=0; x<(width+2); x++) {
                        if (!x && !y)                                   // upper left corner
                                cg[0] = ul;
                        else if (!x && y==width+1)                      // lower left corner
                                cg[size-(width+2)] = dl;
                        else if (!x)                                    // left border
                                cg[y*(width+2)] = lefts[y-1];
                        else if (!y && x==width+1)                      // upper right corner
                                cg[width+1] = ur;
                        else if (x==width+1 && y==width+1)              // lower right corner
                                cg[size-1] = dr;
                        else if (y==width+1)                            // lower border
                                cg[size-(width+2)+x] = downs[x-1];
                        else if (!y)                                    // upper border
                                cg[x] = ups[x-1];
                        else if (x==width+1)                            // right border
                                cg[(y+1)*(width+2)-1] = rights[y-1];
                        else                                            // (inside) - local_grid values
                                cg[y*(width+2)+x] = g[(y-1)*width+(x-1)];
                }
        }

        /* Now update the grid */
        for (int y=1; y<=width; y++) {
                for (int x=1; x<=width; x++) {
                        /* Create sum of the neighbours */
                        int s = cg[(y-1)*(width+2)+x-1] + // up left
                                cg[(y-1)*(width+2)+x] +   // up
                                cg[(y-1)*(width+2)+x+1] + // up right
                                cg[y*(width+2)+x-1] +     // left
                                cg[y*(width+2)+x+1] +     // right
                                cg[(y+1)*(width+2)+x-1] + // down left
                                cg[(y+1)*(width+2)+x] +   // down
                                cg[(y+1)*(width+2)+x+1];  // down right

                        /* Game of Life rules */
                        if ( s<2 || s>3 || (s==2 && !cg[y*(width+2)+x]))
                                g[(y-1)*width+(x-1)] = 0; // Cell dies
                        else
                                g[(y-1)*width+(x-1)] = 1; // Cell lives
                }
        }

        /* Free the pointers */
        free(cg);
}


void transform_for_distribution(int grid[TOTAL_GRID_SIZE], int edge_length) {
        /*
              [ 0  1  2  3           box0     box1      box2         box3
                4  5  6  7    >>> [0 1 4 5  2 3 6 7  8 9 12 13  10 11 14 15 ]
                8  9 10 11
               12 13 14 15 ]

                Goal: Prepare the normal grid (left) for easy equal distribution of subgrids (right) with MPI_Scatter.
        */

        /* Create a copy of the grid */
        int copy_grid[TOTAL_GRID_SIZE] = {0};
        memcpy(copy_grid, grid, sizeof(int)*TOTAL_GRID_SIZE);

        /* Calculate the new indices for every element */
        for (int i=0; i<TOTAL_GRID_SIZE; i++) {
                int row = i/GRID_WIDTH;
                int col = i%GRID_WIDTH;
                int ng_row = row/edge_length;
                int ng_col = col/edge_length;

                /* Get the offset of the box */
                int box_offset = edge_length*(ng_row*GRID_WIDTH + ng_col*edge_length);

                /* Get the offset inside the box */
                int inbox_offset = edge_length*(row%edge_length)+(col%edge_length);

                grid[box_offset + inbox_offset] = copy_grid[i];
        }
}


void transform_from_distribution(int grid[TOTAL_GRID_SIZE], int edge_length) {
        /*
                                                             [ 0  1  4  5
            box0     box1      box2         box3               2  3  6  7
         [ 0 1 2 3  4 5 6 7  8 9 10 11  12 13 14 15 ]  >>>     8  9 12 13
                                                              10 11 14 15 ]

                Goal: Merge the concatenated subgrids from MPI_Gather (left) to the normal grid (right).
        */

        /* Create a copy of the grid */
        int copy_grid[TOTAL_GRID_SIZE] = {0};
        memcpy(copy_grid, grid, sizeof(int)*TOTAL_GRID_SIZE);

        int box_size = edge_length*edge_length;
        int boxes_per_row = GRID_WIDTH / edge_length;

        /* Calculate the original position of each cell in the grid */
        for (int i=0; i<TOTAL_GRID_SIZE; i++) {
                int box_index = i / box_size; // Get the index of the box
                int box_col = box_index % boxes_per_row; // Determine row and column of that box in the grid
                int box_row = box_index / boxes_per_row;

                int inbox_offset = i % box_size; // Get the offset (row and col) inside the box
                int inbox_col = inbox_offset % edge_length;
                int inbox_row = inbox_offset / edge_length;

                // Combine the box and inbox offsets to calculate original index
                int new_col = box_col * edge_length + inbox_col;
                int new_row = (box_row * edge_length + inbox_row) * GRID_WIDTH;

                grid[new_row + new_col] = copy_grid[i];
        }
}

void draw_grid(int grid[TOTAL_GRID_SIZE], int edge_length) {

        fprintf(stdout, S_TOPLEFT);
        for (int y=0; y<GRID_WIDTH; y++){
                for(int x=0; x<GRID_WIDTH; x++)
                        if (COLOR_SUB_GRIDS) {
                                /* Get corresponding processor index for this pixel */
                                int pi = ((int)(y/edge_length))*(GRID_WIDTH/edge_length)+((int)(x/edge_length));
                                fprintf(stdout, "%s  %s", grid[y*GRID_WIDTH+x] ? C_B_BLACK : ARR_COLORS[pi%NUM_COLORS], ARR_COLORS[pi%NUM_COLORS]);
                        } else
                                fprintf(stdout, "%s  %s", grid[y*GRID_WIDTH+x] ? C_B_BLACK : C_B_WHITE, C_B_WHITE);
                fprintf(stdout, "\n");
        }
        fprintf(stdout, C_RST);
        fflush(stdout);
}


void draw_local_grid(int *local_grid, int edge_length) {

        /* Sanity check of the grid size for the LED HAT */
        if (edge_length != 8)
                return;

        /* INCOMPLETE - adapt to the cluster */

        /* Stub for drawing the grid on a Raspberry Pi LED HAT */
        for (int y=0; y<edge_length; y++)
                for(int x=0; x<edge_length; x++)
                        // Replace the printf with an implementation for killing or setting the pixel at x,y.
                        printf("%d",local_grid[y*edge_length+x]);
}


void get_neighbour_ids(int proc_ids[8], int rank, int n_procs) {
        /* Get processors per row - same as GRID_WIDTH / local_edge_length */
        const int ppl = (int)sqrt(n_procs);

        int row = rank/ppl;
        int col = rank%ppl;

        /* Get the eight neighbours with wrap-around.
         * Straight neighbours are distinguished by no wrap-around or wrap-around:
         *
         *            (no wrap-around)? <> : <wrap-around>;
         *
         * Diagonal relations have four scenarios: normal, wrap around x, wrap around y and wrap around x & y.
         * The condition structure is as follows:
         *
         *            (normal condition)? <> : ( (one edge case)? <> : ( (other edge case)? <> : <corner-case> ) );
         */
        proc_ids[0] = (row && col)? rank-ppl-1 : (row? rank-1: (col? (ppl*(ppl-1)+col-1) : n_procs-1)); // up left
        proc_ids[1] = (row)? rank-ppl: (ppl*(ppl-1)+col); // up
        proc_ids[2] = (row && (col<(ppl-1)))? rank-ppl+1 : (row? (row-1)*ppl : ((col<(ppl-1))? ppl*(ppl-1)+col+1  : ppl*(ppl-1)));  // up right
        proc_ids[3] = (col)? rank-1 : rank+ppl-1 ; // left
        proc_ids[4] = (col<(ppl-1))? rank+1 : rank-ppl+1; // right
        proc_ids[5] = (col && (row<(ppl-1)))? rank+ppl-1 : (col? col-1 : ((row<(ppl-1))? ((row+2)*ppl)-1  : ppl-1 )); // down left
        proc_ids[6] = (row<(ppl-1))? rank+ppl : col; // down
        proc_ids[7] = (row<(ppl-1) && col<(ppl-1))? rank+ppl+1 : ((row<(ppl-1))? rank+1 : ((col<(ppl-1))? col+1: 0));// down right
}
