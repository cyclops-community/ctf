#!/bin/bash
REL_SCRIPT_DIR=$(dirname $0)
SCRIPT=$(${REL_SCRIPT_DIR}/manual_readlink.sh $0)
SCRIPT_DIR=$(dirname $SCRIPT)
DIR=$(pwd)
cd $(dirname $1)
FNAME=$(basename $1)
FDIR=$(pwd)
FULLFNAME="${FDIR}/${FNAME}"
if grep -Fxq "$FULLFNAME" $SCRIPT_DIR/visited_list.txt
then
  exit 0
else
  echo $FULLFNAME >> $SCRIPT_DIR/visited_list.txt
  TMP_FILE="${FNAME}.tmp.concat"
  cp $FNAME $TMP_FILE
  sed -i -e 's/#include "\(.*\)"/include \1/g' $TMP_FILE
  awk '
  $1=="include" && NF>=2 {
     system("'$SCRIPT' " $2)
     next
  }
  {print}' "$TMP_FILE"
  rm $TMP_FILE
fi
