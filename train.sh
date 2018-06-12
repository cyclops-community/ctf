NUM_PROCESS="1"
TRAIN_TIME="5"
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n|--num_process)
    NUM_PROCESS="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--train_time)
    TRAIN_TIME="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo NUM PROCESS  = "${NUM_PROCESS}"
echo TRAIN TIME  = "${TRAIN_TIME}"
echo "Compiling"
make model_trainer # compile the program
rm ./src/shared/data/* # clean up the training log files
rm ./src/shared/plot/* # clean up the model plots
mpirun -n ${NUM_PROCESS} ./bin/model_trainer -time ${TRAIN_TIME} -dump # Execute the program
./plot.sh
