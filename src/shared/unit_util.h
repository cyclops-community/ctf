/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __UNIT_UTIL_H__
#define __UNIT_UTIL_H__

#include <algorithm>
#include <string.h>
#include <string>
#include <stdlib.h>

char* getCmdOption(char ** begin, 
                   char ** end, 
                   const std::string & option);

void read_param_file(char const *       fname,
                     int const          myRank,
                     char ***           argv,
                     int *              argc);




#endif
