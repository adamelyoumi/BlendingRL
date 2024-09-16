cd "C:\Users\adame\OneDrive\Bureau\CODE\BlendingRL\aws"
Set-Variable remoteIP "ubuntu@ec2-54-147-166-19.compute-1.amazonaws.com"

ssh -i ".\bp2.pem" $remoteIP -t "
    cd /opt;
    sudo wget https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz;
    sudo tar xvfz gurobi11.0.3_linux64.tar.gz;
    sudo rm gurobi11.0.3_linux64.tar.gz;
    sudo touch gurobi/gurobi.lic;
    export GRB_LICENSE_FILE=/home/ubuntu/gurobi/gurobi.lic;
    export GUROBI_HOME="/opt/gurobi1103/linux64";
    export PATH="${PATH}:${GUROBI_HOME}/bin";
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib";
    cd /home/ubuntu/;
    mkdir bp /home/ubuntu/configs /home/ubuntu/configs/json;
    cd bp;
    mkdir gurobi configs configs/json logs RL_scripts;
    exit;
"

scp -i ".\bp2.pem" "./gurobi.lic" "${remoteIP}:/home/ubuntu/bp/gurobi/"
scp -i ".\bp2.pem" "./reqs.txt" "../pyomo_scripts/pyomo_datagen_simple.py" "../envs.py" "../models.py" "../pyomo_scripts/pyomo_datagen.py" "${remoteIP}:/home/ubuntu/bp/"
scp -i ".\bp2.pem" "../RL_scripts/RL_multitrain" "${remoteIP}:/home/ubuntu/bp/RL_scripts/"
scp -i ".\bp2.pem" "../configs/json/action_sample_base.json" "../configs/json/action_sample_simple.json" "../configs/json/action_sample_simplest.json" "../configs/json/connections_base.json" "../configs/json/connections_simple.json" "../configs/json/connections_simplest.json" "${remoteIP}:/home/ubuntu/configs/json"

$filesToCopy = @()

for ($k = 20; $k -lt 30; $k++) {
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

ssh -i "./bp2.pem" ${remoteIP} -t "
    cd bp;
    pip install -r ./reqs.txt;
    mkdir data data/simple models models/imitation models/simplest;
    python RL_scripts/RL_multitrain.py --configs '[23, 25, 27, 28]' --n_tries 4 --n_timesteps 150000 --layout simplest > out;
"

# ssh -i ".\bp2.pem" -NL 8008:localhost:8008 $remoteIP
# tensorboard --logdir . --port 8008