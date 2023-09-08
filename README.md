# imaging_arsenal
This is Vincent's docker env for general purpose imaging analysis.
By Qing Wang (Vincent)

## Description
This docker env is adapted from the fMRIPrep docker [fMRIPrep-docker](https://github.com/nipreps/fmriprep/blob/master/Dockerfile) (Ubuntu 20.04) , python packages include: 

```
scikit-image=0.19
scikit-learn=1.1
scipy=1.8
seaborn=0.11

datalad=0.16
dipy=1.5
graphviz=3.0
colorclass=2.2
matplotlib=3.5

nibabel=3.2
nilearn=0.9
nipype=1.8
nitime=0.9

attrs=21.4
codecov=2.1
coverage=6.3
curl=7.83
flake8=4.0
git=2.35
h5py=3.6
indexed_gzip=1.6
jinja2=3.1
libxml2=2.9
libxslt=1.1
lockfile=0.12
mkl=2022.1
mkl-service=2.4

nodejs=16
numpy=1.22
packaging=21.3
pandas=1.4
pandoc=2.18
pbr=5.9
pip=22.0
pockets=0.9
psutil=5.9
pydot=1.4
pytest=7.1
pytest-cov=3.0
pytest-env=0.6
pytest-xdist=2.5
pyyaml=6.0
requests=2.27
setuptools=62.3
sphinx=4.5
sphinx_rtd_theme=1.0
svgutils=0.3
toml=0.10
traits=6.3
zlib=1.2
zstd=1.5
```

Additional packages include:

### Conda:

#### adding excel data support.
conda install openpyxl=3.0.10  

### Pip:

#### Adding homepage support [@https://github.com/Vincent-wq/qingwang_vincent.github.io] [In construction]:

including: 

'''apt-get install ruby-full build-essential zlib1g-dev 
[https://jekyllrb.com/docs/installation/ubuntu/]
'''

#### Adding bibliometric support [https://github.com/pybliometrics-dev/pybliometrics]:

pip install pybliometrics

pip install mlens

## Build

```
cd imaging_arsenal\docker

sudo docker build -t vincent_env -f Dockerfile .

```

## Use

for Windows:

```
docker run -it -p 127.0.0.1:8888:8888 -v C:\Users\Vincent\Desktop\scratch:/scratch vincent_env:latest jupyter-lab --notebook-dir=/scratch --ip=0.0.0.0 --no-browser --allow-root
```

For Linux:

```
sudo docker run -it -p 127.0.0.1:8888:8888 -v $HOME/scratch:/scratch vincent_env:latest jupyter-lab --notebook-dir=/scratch --ip=0.0.0.0 --no-browser --allow-root
```