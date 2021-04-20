CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV)

.PHONY: setup jn

setup: requirements.yaml data/boyd_2019/
	mamba env create --force --prefix $(CONDA_ENV) --file requirements.yaml

data/boyd_2019/:
	git clone git@github.com:jcboyd/multi-cell-line.git ../multi-cell-line
	ln -s ../multi-cell-line/cecog_out_propagate_0.5 data/boyd_2019

jn:
	$(CONDA_ACTIVATE); jupyter notebook --notebook-dir=notebooks/
