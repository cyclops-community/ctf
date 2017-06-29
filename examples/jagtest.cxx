#include <ctf.hpp>

int main(){
  int ord = 9;
  std::vector<int> fours;
  std::vector<int> nns;
  for (int i=0; i<ord; i++){
    fours.push_back(4);
    nns.push_back(NS);
  }
  MPI_Init(0,NULL);
  {
    cCTF_World dw(MPI_COMM_WORLD);
    cCTF_Tensor Wolverine(ord, &(fours[0]), &(nns[0]), dw);
    cCTF_Tensor Sabretooth(ord, &(fours[0]), &(nns[0]), dw);
    cCTF_Tensor jagger1st(2, &(fours[0]), &(nns[0]), dw);
    Wolverine["abcdefghijklmnopqrstuvwx"] = jagger1st["hd"] * Sabretooth["abcdefghijklmnopqrstuvwx"];

  }
  MPI_Finalize();
  return 0;
}
