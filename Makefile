CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV)

.PHONY: clean setup jn download_data
all: train

setup: $(CONDA_ENV) data/boyd_2019/
	git config core.hooksPath .githooks

$(CONDA_ENV): requirements.yaml
	mamba env create --force --prefix $(CONDA_ENV) --file requirements.yaml

data/boyd_2019/: data/boyd_2019.dvc
	$(CONDA_ACTIVATE); dvc pull

download_data:
	$(CONDA_ACTIVATE); dvc get https://github.com/jcboyd/multi-cell-line cecog_out_propagate_0.5
	mv cecog_out_propagate_0.5 data/boyd_2019

jupyter:
	$(CONDA_ACTIVATE); export PYTHONPATH=`pwd`:$${PYTHONPATH}; jupyter lab --notebook-dir=notebooks/

train: src/train.nf
	mkdir -p results/boyd_2019
	$(CONDA_ACTIVATE); nextflow src/train.nf --out results/boyd_2019 -resume -profile gpu --gpus 9

fc_train: src/fc_train.nf
	mkdir -p results/boyd_2019
	$(CONDA_ACTIVATE); nextflow src/fc_train.nf --out results/boyd_2019 -resume -profile gpu --gpus 9

clean:
	rm -rf env/
