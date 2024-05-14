SHELL := /bin/bash

clean:
	rm -rf venv

download:
	wget -c -N -O task-oriented-paraphrase-analytics-data.zip https://zenodo.org/records/11191536/files/task-oriented-paraphrase-analytics-data.zip?download=1
	unzip -u task-oriented-paraphrase-analytics-data.zip

install: download
	python3 -m venv venv
	source venv/bin/activate && pip install -r requirements.txt

train:
	source venv/bin/activate && python src/train_classifier.py

predict:
	source venv/bin/activate && python src/predict_tasks.py