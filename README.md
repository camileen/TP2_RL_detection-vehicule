# IAT TP2 : Reinforcement Learning, vehicule dectection

## How to run the model?

Steps:

1. Create a Python venv running the following command: `./app-env.sh`
2. Select the created venv: `source ./.venv/iat-tp2/bin/activate`
3. Install dependances: `pip install -r requirements.txt`
4. Download dataset running the command: `./download-dataset.sh`
5. Finally, run the app: `python3 main.py`

The app might take time to run.
The calculated Q matrix is stored in a JSON file in the ./results/q-tables/ directory.
Same for calculated metrics of performance (./results/metrics/ directory).
