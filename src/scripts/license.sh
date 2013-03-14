#!/bin/sh 

for file in ../*/*.hxx ../*/*.cxx ../*/*.h 
do
  if grep MERCH $file
  then
    vim $file -c ":d22" -c ":%s/\ \*\ SUCH\ DAMAGE\.\ \*\//\/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*\//g" -c ":wq"
  fi
done
