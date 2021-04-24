CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV)

.PHONY: clean setup jn

setup: $(CONDA_ENV) 
	dvc init

$(CONDA_ENV): requirements.yaml
	mamba env create --force --prefix $(CONDA_ENV) --file requirements.yaml

data/boyd_2019/:
	$(CONDA_ACTIVATE); dvc get https://github.com/jcboyd/multi-cell-line cecog_out_propagate_0.5
	mv cecog_out_propagate_0.5 data/boyd_2019

jn:
	$(CONDA_ACTIVATE); jupyter notebook --notebook-dir=notebooks/

clean:
	rm -rf env/
