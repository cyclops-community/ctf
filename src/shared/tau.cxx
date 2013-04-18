#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include "string.h"
#include "assert.h"
#include <iostream>
#include <vector>
#include "tau.h"

#define MAX_NAME_LENGTH 38

int main_argc=-1;
char ** main_argv;
MPI_Comm comm;
double excl_time;
double complete_time;
int set_contxt = 0;
    

class function_timer{
  public:
    char name[MAX_NAME_LENGTH];
    double start_time;
    double start_excl_time;
    double acc_time;
    double acc_excl_time;
    int calls;

    double total_time;
    double total_excl_time;
    int total_calls;

  public: 
    function_timer(char const * name_, 
                   double const start_time_,
                   double const start_excl_time_){
      sprintf(name, "%s", name_);
      start_time = start_time_;
      start_excl_time = start_excl_time_;
      if (strlen(name) > MAX_NAME_LENGTH) {
        printf("function name must be fewer than %d characters\n",MAX_NAME_LENGTH);
        assert(0);
      }
      acc_time = 0.0;
      acc_excl_time = 0.0;
      calls = 0;
    }

    void compute_totals(MPI_Comm comm){ 
      MPI_Allreduce(&acc_time, &total_time, 1, 
                    MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(&acc_excl_time, &total_excl_time, 1, 
                    MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(&calls, &total_calls, 1, 
                    MPI_INT, MPI_SUM, comm);
    }

    bool operator<(function_timer const & w) const {
      return total_time > w.total_time;
    }

    void print(FILE *         output, 
               MPI_Comm const comm, 
               int const      rank,
               int const      np){
      int i;
      if (rank == 0){
        fprintf(output, "%s", name);
        char * space = (char*)malloc(MAX_NAME_LENGTH-strlen(name)+1);
        for (i=0; i<MAX_NAME_LENGTH-(int)strlen(name); i++){
          space[i] = ' ';
        }
        space[i] = '\0';
        fprintf(output, "%s", space);
        fprintf(output,"%5d   %3d.%04d  %3d.%02d  %3d.%04d  %3d.%02d\n",
                total_calls/np,
                (int)(total_time/np),
                ((int)(1000.*(total_time)/np))%1000,
                (int)(100.*(total_time)/complete_time),
                ((int)(10000.*(total_time)/complete_time))%100,
                (int)(total_excl_time/np),
                ((int)(1000.*(total_excl_time)/np))%1000,
                (int)(100.*(total_excl_time)/complete_time),
                ((int)(10000.*(total_excl_time)/complete_time))%100);
      } 
    }
};

bool comp_name(function_timer const & w1, function_timer const & w2) {
  return strcmp(w1.name, w2.name)>0;
}

std::vector<function_timer> function_timers;

CTF_timer::CTF_timer(const char * name){
  int i;
  if (function_timers.size() == 0) {
    original = 1;
    index = 0;
    excl_time = 0.0;
    function_timers.push_back(function_timer(name, MPI_Wtime(), 0.0)); 
  } else {
    for (i=0; i<(int)function_timers.size(); i++){
      if (strcmp(function_timers[i].name, name) == 0){
        /*function_timers[i].start_time = MPI_Wtime();
        function_timers[i].start_excl_time = excl_time;*/
        break;
      }
    }
    index = i;
    original = (index==0);
  }
  if (index == (int)function_timers.size()) {
    function_timers.push_back(function_timer(name, MPI_Wtime(), excl_time)); 
  }
  timer_name = name;
  exited = 0;
}
  
void CTF_timer::start(){
  function_timers[index].start_time = MPI_Wtime();
  function_timers[index].start_excl_time = excl_time;
}

void CTF_timer::stop(){
  double delta_time = MPI_Wtime() - function_timers[index].start_time;
  function_timers[index].acc_time += delta_time;
  function_timers[index].acc_excl_time += delta_time - 
        (excl_time- function_timers[index].start_excl_time); 
  excl_time = function_timers[index].start_excl_time + delta_time;
  function_timers[index].calls++;
  exit();
  exited = 1;
}

CTF_timer::~CTF_timer(){ }

void CTF_timer::exit(){
  if (set_contxt && original && !exited) {
    int rank, np, i, j, p, len_symbols;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    FILE * output;

    char all_symbols[10000];

    if (rank == 0){
      char filename[300];
      char part[300];
      sprintf(filename, "profile.");
      int off;
      if (main_argc != -1){
        for (off=strlen(main_argv[0]); off>=0; off--){
          if (main_argv[0][off-1] == '/') break;
        }
        for (i=0; i<main_argc; i++){
          if (i==0)
            sprintf(filename+strlen(filename), "%s.", main_argv[0]+off);
          else
            sprintf(filename+strlen(filename), "%s.", main_argv[i]);
        }
      } 
      sprintf(filename+strlen(filename), "-p%d.out", np);
      
      
      output = fopen(filename, "w");
      char heading[MAX_NAME_LENGTH+200];
      for (i=0; i<MAX_NAME_LENGTH; i++){
        part[i] = ' ';
      }
      part[i] = '\0';
      sprintf(heading,"%s",part);
      //sprintf(part,"calls   total sec   exclusive sec\n");
      sprintf(part,"       inclusive         exclusive\n");
      strcat(heading,part);
      fprintf(output, "%s", heading);
      for (i=0; i<MAX_NAME_LENGTH; i++){
        part[i] = ' ';
      }
      part[i] = '\0';
      sprintf(heading,"%s",part);
      sprintf(part, "calls        sec       %%"); 
      strcat(heading,part);
      sprintf(part, "       sec       %%\n"); 
      strcat(heading,part);
      fprintf(output, "%s", heading);

      len_symbols = 0;
      for (i=0; i<(int)function_timers.size(); i++){
        sprintf(all_symbols+len_symbols, "%s", function_timers[i].name);
        len_symbols += strlen(function_timers[i].name)+1;
      }
    }
    if (np > 1){
      for (p=0; p<np; p++){
        if (rank == p){
          MPI_Send(&len_symbols, 1, MPI_INT, (p+1)%np, 1, comm);
          MPI_Send(all_symbols, len_symbols, MPI_CHAR, (p+1)%np, 2, comm);
        }
        if (rank == (p+1)%np){
          MPI_Status stat;
          MPI_Recv(&len_symbols, 1, MPI_INT, p, 1, comm, &stat);
          MPI_Recv(all_symbols, len_symbols, MPI_CHAR, p, 2, comm, &stat);
          for (i=0; i<(int)function_timers.size(); i++){
            j=0;
            while (j<len_symbols && strcmp(all_symbols+j, function_timers[i].name) != 0){
              j+=strlen(all_symbols+j)+1;
            }
            
            if (j>=len_symbols){
              sprintf(all_symbols+len_symbols, "%s", function_timers[i].name);
              len_symbols += strlen(function_timers[i].name)+1;
            }
          }
        }
      }
      MPI_Bcast(all_symbols, len_symbols, MPI_CHAR, 0, comm);
      j=0;
      while (j<len_symbols){
        CTF_timer t(all_symbols+j);
        j+=strlen(all_symbols+j)+1;
      }
    }

    std::sort(function_timers.begin(), function_timers.end(),comp_name);
    for (i=0; i<(int)function_timers.size(); i++){
      function_timers[i].compute_totals(comm);
    }
    std::sort(function_timers.begin(), function_timers.end());
    complete_time = function_timers[0].total_time;
    for (i=0; i<(int)function_timers.size(); i++){
      function_timers[i].print(output,comm,rank,np);
    }
    
    if (rank == 0){
      fclose(output);
    } 
    
  }
}

void CTF_set_main_args(int argc, char ** argv){
  main_argv = argv;
  main_argc = argc;
}

void CTF_set_context(int const ctxt){
  set_contxt = 1;
  if (ctxt == 0) comm = MPI_COMM_WORLD;
  else comm = ctxt;
}
