PYTHON ?= python3

.PHONY: setup data eda forecast anomaly evaluate analytics evidence dashboard test all clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data --config config.yaml

eda: data
	$(PYTHON) scripts/eda.py --config config.yaml

forecast: data
	$(PYTHON) -m src.forecast --config config.yaml

anomaly: forecast
	$(PYTHON) -m src.anomaly --config config.yaml

analytics:
	$(PYTHON) -m src.sql_analytics

evidence:
	$(PYTHON) -m src.evidence --config config.yaml

evaluate: anomaly
	$(PYTHON) -m src.capacity --config config.yaml
	$(PYTHON) -m src.evaluate --config config.yaml
	$(PYTHON) -m src.sql_analytics
	$(PYTHON) -m src.evidence --config config.yaml
	$(PYTHON) scripts/generate_readme_results.py

dashboard:
	streamlit run app/dashboard.py

test:
	$(PYTHON) -m pytest -q

all: data eda forecast anomaly evaluate test

clean:
	rm -rf data/processed/* results/* figures/*
	touch data/processed/.gitkeep results/.gitkeep figures/.gitkeep

