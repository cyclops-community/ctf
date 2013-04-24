/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include "unit_util.h"
#include "comm.h"
#include "util.h"


/**
 * \brief pulls out value associated with matched string
 * \param[in] begin start of array of strings
 * \param[in] end ending of array of strings
 * \param[in] option string ot match
 * \return value of the string after the one matched
 */
char* getCmdOption(char ** begin, 
                   char ** end, 
                   const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

/**
 * \brief reads parameter file into args format
 * \param[in] fname file name of the parameter ifle to read
 * \param[in] myRank processor rank
 * \param[out] argv file parsed by string
 * \param[int] argc number of strings 
 */
void read_param_file(char const *       fname,
                     int const          myRank,
                     char ***           argv,
                     int *              argc){
  int tot_sz, i, in_num, ret;
  char * iter_str, * serial_str = NULL, ** input_str = NULL;
  if (myRank == 0) {
    tot_sz = 0;
    input_str = (char**)CTF_alloc(1000*sizeof(char*));
    char add_str[1000];
    FILE * inFile;
    printf("Opening input file %s\n", fname);
    inFile = fopen(fname,"r");
    assert(inFile != NULL);
    in_num=0;
    while (!feof(inFile)){
      ret = fscanf(inFile, "%s", add_str);
      assert(ret!=EOF);
      input_str[in_num] = (char*)CTF_alloc(strlen(add_str)+1);
      tot_sz += strlen(add_str)+1;
      strcpy(input_str[in_num], add_str);
      in_num++;
    }
    serial_str = (char*)CTF_alloc(tot_sz);
    iter_str = serial_str;
    for (i=0; i<in_num; i++){
      strcpy(iter_str, input_str[i]);
      iter_str += strlen(input_str[i])+1;
    }
    //assert(0==fclose(inFile));
  } 
  MPI_Bcast(&in_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tot_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (myRank >0) serial_str = (char*)CTF_alloc(tot_sz*sizeof(char));
  MPI_Bcast(serial_str, tot_sz, MPI_CHAR, 0, MPI_COMM_WORLD);
  if (myRank >0){
    iter_str = serial_str;
    input_str = (char**)CTF_alloc(in_num*sizeof(char*));
    for (i=0; i<in_num; i++){
      input_str[i] = (char*)CTF_alloc(strlen(iter_str)+2);
      strcpy(input_str[i], iter_str);
      iter_str += strlen(input_str[i])+1;
    }
  }
  (*argv) = input_str;
  (*argc) = in_num;
  CTF_free(serial_str);
}
