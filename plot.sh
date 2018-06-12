for file in ./src/shared/data/*; do
    python3 ./src/shared/plot_model.py $file
done
