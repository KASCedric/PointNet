# PointNet


## How to set up (On Ubuntu) ? ##

### Step 1: Update your repositories
```
sudo apt-get update
```

### Step 2: Install pip for Python 3
```
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
sudo apt install python3-pip
```

### Step 3: Use pip to install virtualenv
```
sudo pip3 install virtualenv 
```

### Step 4: Launch your Python 3 virtual environment, here the name of my virtual environment will be venv
```
virtualenv -p python3 venv
```

### Step 5: Activate your new Python 3 environment
```
source venv/bin/activate
```

### This commands will show you what is going on: the python executable you are using is now located inside your virtualenv repository
```
which python 
python --version
```

### Step 6: Install the requirements for the project
```
pip install -r requirements.txt
```

### Step 7: Do whatever you want:
```
...
```

### Step 8: done? leave the virtual environment
```
deactivate
```
