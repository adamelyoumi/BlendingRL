cd C:\Users\adame\Downloads
ssh -i "bpkey.pem" ec2-user@ec2-3-88-68-57.compute-1.amazonaws.com

cd /opt
sudo yum install python3-pip
pip install numpy pandas Pyomo
sudo wget https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz
sudo tar xvfz gurobi11.0.3_linux64.tar.gz
sudo rm gurobi11.0.3_linux64.tar.gz
sudo mkdir gurobi
sudo touch gurobi/gurobi.lic
export GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
export GUROBI_HOME="/opt/gurobi1103/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

cd /home/ec2-user/
mkdir bp
cd bp
touch datagen_simple.py
mkdir data
mkdir data/simple

