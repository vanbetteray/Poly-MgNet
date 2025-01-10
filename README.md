## Requirements
Currently running with 
  python 3.6 with CUDA 11.0
### Packages
```bash
pip3 install --user --upgrade pip
pip3 install --r requirements

```

## Settings
Specify settings in configs.json. Some arguments need to be updated in \texttt{train_routine_MgNet_poly.py} or \texttt{train_routine_MgNet_quad.py}.

## Run
For $$\text{MgNet}^{q2}$$ run
```bash
python3 train_routine_MgNet_poly.py
```
For  $$\text{MgNet}^{g4}$$,  $$\text{MgNet}^{g6}$$ or $$\text{MgNet}^{g8}$$ run
```bash
python3 train_routine_MgNet_quad.py
```