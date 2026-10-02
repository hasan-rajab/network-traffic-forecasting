PYTHON ?= python3
.PHONY: setup data eda forecast anomaly evaluate dashboard test all
setup:
	$(PYTHON) -m pip install -r requirements.txt
data:
	$(PYTHON) -m src.data
eda: data
	$(PYTHON) scripts/eda.py
forecast:
	@echo "Forecast training executes after real data is available; implementation is intentionally gated by measured-data workflow."
anomaly:
	@echo "Anomaly evaluation executes after real held-out forecasts are available."
evaluate:
	$(PYTHON) scripts/generate_readme_results.py
dashboard:
	streamlit run app/dashboard.py
test:
	$(PYTHON) -m pytest -q
all: data eda test evaluate
