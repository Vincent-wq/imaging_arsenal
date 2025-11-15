# imaging_arsenal
This is Vincent's docker env for general purpose imaging analysis.
By Qing Wang (Vincent)

## Description
This docker env is adapted from the fMRIPrep docker [fMRIPrep-docker](https://github.com/nipreps/fmriprep/blob/master/Dockerfile) (Ubuntu 20.04) , python packages include: 

```
scikit-image=0.19
scikit-learn=1.1
scipy=1.13.0   (updated from 1.8.0)
seaborn=0.13.2   (updated from 0.11)

datalad=0.16
dipy=1.5
graphviz=3.0
colorclass=2.2
matplotlib=3.9.2 (updated from 3.5)

nibabel=3.2
nilearn=0.10.4
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

#### Adding imaging tools
pip install [surfplot](https://pypi.org/project/surfplot/)

pip install neuromaps==0.0.5

or  Github install [brainmaps](https://github.com/netneurolab/neuromaps)

pip install dowhy
pip install pydot
pip install econml

(lib for causal inference)

pip install py-irt

(lib for item respose theory)

pip install pingouin

(lib for general and fast statitical tests)

pip install factor-analyzer

(lib for EFA/CFA etc.)

#### Adding time series support

pip install nolds

(lib for non-linear dynamics modeling)

pip install antropy

(lib for entropy based measures)

pip install hurst

(lib for Hurst Exponent)

pip install hmmlearn

(lib for Hidden Markov Models (HMM))

pip install PyEMD

(lib for computing Earth Mover's Distance)

pip install EMD-signal

(lib for Empirical Mode Decomposition (EMD))

pip install pyRQA

(lib for Recurrence Quantification Analysis (RQA))

## adding visualization libs

pip install 'pyvista[all,trame]' jupyterlab
pip install "trame_jupyter_extension<2"   (for jupyter-lab 3.x)

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

with X11 support 

```
docker run -it --env DISPLAY=:99 -p 127.0.0.1:8888:8888 -v C:\Users\Vincent\Desktop\scratch:/scratch vincent_env:241216 bash -c "Xvfb :99 -screen 0 1920x1080x24 & jupyter-lab --notebook-dir=/scratch --ip=0.0.0.0 --no-browser --allow-root"
```

For Linux:

```
sudo docker run -it -p 127.0.0.1:8888:8888 -v $HOME/scratch:/scratch vincent_env:latest jupyter-lab --notebook-dir=/scratch --ip=0.0.0.0 --no-browser --allow-root
```

## Container backup

docker commit <container_id> <image_name>:<tag>

docker save -o <backup_image.tar> <image_name>:<tag>

docker load -i <backup_image.tar>
