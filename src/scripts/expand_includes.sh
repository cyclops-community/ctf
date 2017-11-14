#!/bin/bash
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
touch ${SCRIPT_DIR}/visited_list.txt
$SCRIPT_DIR/recursive_expand_includes.sh $SCRIPT_DIR/../../include/ctf.hpp &> $SCRIPT_DIR/../../include/ctf_all.hpp
rm -f ${SCRIPT_DIR}/visited_list.txt
