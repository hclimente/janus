CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV)

.PHONY: clean setup jn download_data

setup: $(CONDA_ENV) data/boyd_2019/

$(CONDA_ENV): requirements.yaml
	git config core.hooksPath .githooks
	mamba env create --force --prefix $(CONDA_ENV) --file requirements.yaml

data/boyd_2019/: data/boyd_2019.dvc
	$(CONDA_ACTIVATE); dvc pull

download_data:
	$(CONDA_ACTIVATE); dvc get https://github.com/jcboyd/multi-cell-line cecog_out_propagate_0.5
	mv cecog_out_propagate_0.5 data/boyd_2019

jn:
	$(CONDA_ACTIVATE); jupyter notebook --notebook-dir=notebooks/

clean:
	rm -rf env/
