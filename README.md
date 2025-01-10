## Requirements
currently running with 
  python 3.6 with CUDA 11.0
  python 3.9 with CUDA 12.1
### Packages
```bash
pip3 install --user --upgrade pip
pip3 install --r requirements

```

### Path

The following warning occurs if the path is not appropriately set:
```
  The scripts convert-caffe2-to-onnx and convert-onnx-to-caffe2 are installed in '/home/user/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
```

Executing the following command adds the directory for needed binaries to
the path:

```
PATH=~/.local/bin:$PATH
```

To check the variable, run `echo $PATH` and compare with
`echo /home/user/.local/bin:/home/user/bin:/usr/local/bin:/usr/bin:/bin`

## Run

```bash
python3 train_routine_MgNet_poly.py
```
