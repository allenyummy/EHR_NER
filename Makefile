.PHONY: clean

help:
	@echo "usage: make [argument]"
	@echo " "
	@echo "arguments:"
	@echo "  test        :  conduct test functions"
	@echo "  freeze      :  freeze packages and export to requirements.txt"
	@echo "  build       :  build a docker image"
	@echo "  run_sl      :  run EHR_NER as a sequence labeing (sl) task"
	@echo "  run_simqasl :  run EHR_NER as a simple question answering (simqasl) task"
	@echo "  demo        :  run a local demo"

clean:
	find . -name '.pytest_cache' -type d -exec rm -rf {} +
	find . -name '__pycache__' -type d -exec rm -rf {} +

test:
	PYTHONPATH=./ pytest --log-cli-level=debug --cov=train/ --cov=utils/datasets/ --cov=configs/ --cov=tests/
test_model_pred:
	PYTHONPATH=./ pytest -rx tests/test_model_pred/ --log-cli-level=info --cov=./
freeze:
	poetry export --without-hashes -f requirements.txt --output requirements.txt
build:
	docker build . -t ehr_ner:0.1.1-rc
run_sl:
	PYTHONPATH=./ python train/run_ner.py sl
run_simqasl:
	PYTHONPATH=./ python train/run_ner.py simqasl
run_qasl:
	PYTHONPATH=./ python train/run_ner.py qasl
run_mrc:
	PYTHONPATH=./ python train/run_ner.py mrc
demo:
	PYTHONPATH=./ python demo/app.py
