python experiment.py --network dnn --amount 50000 --first_layers 4 --second_layers 4 -f 0.5 -n 1536 --name "4-4-1536neurons"
python experiment.py --network dnn --amount 50000 --first_layers 4 --second_layers 4 -f 0.5 -n 2048 --name "4-4-2048neurons"
python experiment.py --network dnn --amount 50000 --first_layers 4 --second_layers 4 -f 0.5 -n 3072 --name "4-4-3072neurons"

python experiment.py --network mlp --amount 50000 --first_layers 4 --second_layers 4 -f 0.5 -n 1536 --name "4-4-1536neurons"
python experiment.py --network mlp --amount 50000 --first_layers 4 --second_layers 4 -f 0.5 -n 2048 --name "4-4-2048neurons"
python experiment.py --network mlp --amount 50000 --first_layers 4 --second_layers 4 -f 0.5 -n 3072 --name "4-4-3072neurons"
