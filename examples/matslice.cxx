#include <ctf.hpp>
using namespace CTF;
using namespace std;


void print_time_information(int level, ScheduleTimer st){
    printf("For level %d:\n", level);
    printf("Schedule comm down: %lf\n", st.comm_down_time);
    printf("Schedule execute: %lf\n", st.exec_time);
    printf("Schedule imbalance, wall: %lf\n", st.imbalance_wall_time);
    printf("Schedule imbalance, accum: %lf\n", st.imbalance_acuum_time);
    printf("Schedule comm up: %lf\n", st.comm_up_time);
    printf("Schedule total: %lf\n", st.total_time);
}

int main(int argc, char ** argv){
	int rank, np;
	int n = 2, k = 2;
	MPI_Init(&argc, &argv);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 	MPI_Comm_size(MPI_COMM_WORLD, &np);

 	World dw(argc, argv);

 	bool use_projection = false;
 	int num_levels = 1;

 	for (int i = 0; i < num_levels; i++){
 		int count = pow(2, i);

 		vector<Matrix<> > S;
 		vector<Matrix<> > K;
 		vector<Matrix<> > B;
 		vector<Matrix<> > C;

 		vector<Matrix<> > ref_S;
 		vector<Matrix<> > ref_K;
 		vector<Matrix<> > ref_B;
 		vector<Matrix<> > ref_C;

 		for (int j = 0; j < count; j++){
 			// C[j] = S[j]K[i]B[j]

 			if (use_projection){

 			}

 			Matrix<> Sj(n/count, k, NS, dw);
 			Matrix<> Kj(k, n/count, NS, dw);
 			Matrix<> Bj(n/count, k, NS, dw);
 			Matrix<> Cj(n/count, k, NS, dw);

 			Sj.fill_random(0.0, 1.0);
 			Kj.fill_random(0.0, 1.0);
 			Bj.fill_random(0.0, 1.0);
 			Cj.fill_random(0.0, 1.0);

 			Matrix<> ref_Sj(n/count, k, NS, dw);
 			Matrix<> ref_Kj(k, n/count, NS, dw);
 			Matrix<> ref_Bj(n/count, k, NS, dw);
 			Matrix<> ref_Cj(n/count, k, NS, dw);

 			/* copy initial values to reference matrices */
 			ref_Sj["ij"] = Sj["ij"];
 			ref_Kj["ij"] = Kj["ij"];
 			ref_Bj["ij"] = Bj["ij"];
 			ref_Cj["ij"] = Cj["ij"];

 			S.push_back(Sj);
 			K.push_back(Kj);
 			B.push_back(Bj);
 			C.push_back(Cj);

 			ref_S.push_back(ref_Sj);
 			ref_K.push_back(ref_Kj);
 			ref_B.push_back(ref_Bj);
 			ref_C.push_back(ref_Cj);
 		}

 		Schedule sched(&dw);
 		sched.record();
 		
 		for (int j = 0; j < count; j++){
 			C[j]["il"] = S[j]["ij"] * K[j]["jk"] * B[j]["kl"];
 		}
 				
 		ScheduleTimer schedule_time = sched.execute();

 		for (int j = 0; j < count; j++){	
 			ref_C[j]["il"] = ref_S[j]["ij"] * ref_K[j]["jk"] * ref_B[j]["kl"];
 			//ref_C[j].print();
 			//C[j].print();
 			ref_C[j]["il"] -= C[j]["il"];
 			printf("Error: %f\n", ref_C[j].norm2());
 		}

 		if (rank == 0) {
 			print_time_information(i, schedule_time);
		}
 	}
	MPI_Finalize();
	return 0;
}
