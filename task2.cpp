#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include "mpi.h"
#define TARGET 16.0/135.0
#define _USE_MATH_DEFINES


double get_sum(int64_t N, double volume, int size) {
    double tmp_vol = 0.0;
    double x, y, z;
    for(int i = 0; i < N; ++i)
        {
            x = -1.0 + 2.0 * (double)rand()/RAND_MAX;
            y = -1.0 + 2.0 * (double)rand()/RAND_MAX;
            z = -2.0 + 4.0 * (double)rand()/RAND_MAX;
            if (fabs(x) + fabs(y) <= 1)
            {
            tmp_vol += volume*x*x*y*y*z*z/(N*size);
            }
	}
    return tmp_vol;
}
int main(int argc, char *argv[]){
    int rank, size;
    int iter = 1, stop = 1;
    double x, y, z, volume;
    double eps = 0.0, buf = 0.0, result_vol,
           s_time, tmp_vol = 0.0, tmp_vol_allp = 0.0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    volume =  2.0*2.0*4.0;
    double target_eps = atof(argv[1]);
    int64_t N = (int64_t) 9000.0 * (1.0 / (target_eps*size));
    srand(2*rank + 1);
    s_time = MPI_Wtime();
    while(stop) {
        tmp_vol = get_sum(N, volume, size);
        MPI_Allreduce(&tmp_vol, &tmp_vol_allp, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        eps = std::abs(TARGET - tmp_vol_allp/iter);
        if (eps < target_eps) {
            stop = 0;
        }
	iter++;
    }
    result_vol = tmp_vol_allp/(iter - 1);
    double e_time = MPI_Wtime();
    double res_time;
    double tmp_time = e_time - s_time;
    MPI_Reduce(&tmp_time, &res_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout<<
            "Eps: "<<eps<<std::endl<<
            "N: "<<size * N * iter<<std::endl<<
            "Time: "<<res_time<<std::endl<<
            "Integral: "<<result_vol<<std::endl<<
            "True: "<<TARGET<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
