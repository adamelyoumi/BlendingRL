cd "C:\Users\adame\OneDrive\Bureau\CODE\BlendingRL\aws"
Set-Variable remoteIP "ubuntu@ec2-44-223-37-152.compute-1.amazonaws.com"

ssh -i ".\bp2.pem" $remoteIP -t "    
    export GRB_LICENSE_FILE=/home/ubuntu/bp/gurobi/gurobi.lic;
    export GUROBI_HOME=/opt/gurobi1103/linux64;
    export PATH=${PATH}:${GUROBI_HOME}/bin;
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib;
    
    cd /opt;
    sudo wget https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz;
    sudo tar xvfz gurobi11.0.3_linux64.tar.gz;
    sudo rm gurobi11.0.3_linux64.tar.gz;
    sudo touch gurobi/gurobi.lic;
    cd /home/ubuntu/;
    mkdir bp;
    cd bp;
    mkdir gurobi configs configs/json logs RL_scripts pyomo_scripts;
    git clone https://github.com/kzl/decision-transformer.git
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    exit;
"

scp -i ".\bp2.pem" "./gurobi.lic" "${remoteIP}:/home/ubuntu/bp/gurobi/"
scp -i ".\bp2.pem" "./reqs.txt" "../envs.py" "../models.py" "../utils.py" "${remoteIP}:/home/ubuntu/bp/"
scp -i ".\bp2.pem" "../pyomo_scripts/pyomo_datagen_simple.py" "../pyomo_scripts/pyomo_datagen.py" "../pyomo_scripts/solver_function.py" "${remoteIP}:/home/ubuntu/bp/pyomo_scripts/"
scp -i ".\bp2.pem" "../RL_scripts/RL_multitrain.py" "${remoteIP}:/home/ubuntu/bp/RL_scripts/"
scp -i ".\bp2.pem" "../configs/json/action_sample_base.json" "../configs/json/action_sample_simple.json" "../configs/json/action_sample_simplest.json" "../configs/json/connections_base.json" "../configs/json/connections_simple.json" "../configs/json/connections_simplest.json" "${remoteIP}:/home/ubuntu/bp/configs/json"

$filesToCopy = @()

for ($k = 40; $k -lt 99; $k++) {
    $filePath = "../configs/$k.yaml"
    if (Test-Path $filePath) {
        $filesToCopy += $filePath
    }
}

if ($filesToCopy.Count -gt 0) {
    $scpCommand = "scp -i '.\bp2.pem' $($filesToCopy -join ' ') '${remoteIP}:/home/ubuntu/bp/configs'"
    Invoke-Expression $scpCommand
} else {
    Write-Host "No files to copy."
}

scp -i ".\bp2.pem" "..\decision-transformer\gym\decision_transformer\evaluation\evaluate_episodes.py" "${remoteIP}:/home/ubuntu/bp/decision-transformer/gym/decision_transformer/evaluation" 
scp -i ".\bp2.pem" "..\decision-transformer\gym\decision_transformer\models\trajectory_gpt2.py" "${remoteIP}:/home/ubuntu/bp/decision-transformer/gym/decision_transformer/models/trajectory_gpt2.py"
scp -i ".\bp2.pem" "..\decision-transformer\gym\experiment.py" "${remoteIP}:/home/ubuntu/bp/decision-transformer/gym/experiment.py" 
scp -i ".\bp2.pem" "..\decision-transformer\gym\data\blend-medium-v4.pkl" "${remoteIP}:/home/ubuntu/bp/decision-transformer/gym/data" 

ssh -i "./bp2.pem" ${remoteIP} -t "
    cd bp;
    pip install -r ./reqs.txt;
    mkdir data data/simple models models/imitation models/simplest;
    python RL_scripts/RL_multitrain.py --configs '[35, 36, 37, 38, 39, 40]' --n_tries 1 --n_timesteps 10000 --layout simplest;
"

# ssh -i ".\bp2.pem" -NL 8008:localhost:8008 $remoteIP
# tensorboard --logdir ./logs/simple --port 8008
# python RL_scripts/RL_multitrain.py --configs '[55, 56, 57, 58, 59, 60, 61, 62]' --n_tries 2 --n_timesteps 100000 --layout simple;
# python ./decision-transformer/gym/experiment.py --env blend --dataset medium --dataset_version v4 --env_type simple --model_type dt --batch_size 128 --max_iters 50 -lr 0.005