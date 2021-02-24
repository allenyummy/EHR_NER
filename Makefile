test:
	PYTHONPATH=./ pytest --log-cli-level=debug --cov=train/ --cov=utils/datasets/ --cov=configs/ --cov=tests/
test_model_pred:
	PYTHONPATH=./::../seqhelper pytest -rx tests/test_model_pred/ --log-cli-level=info --cov=./
freeze:
	poetry export --without-hashes -f requirements.txt --output requirements.txt
build:
	docker build . -t ehr_ner:0.1.0
run_sl:
	PYTHONPATH=./ python train/run_ner.py sl
run_simqasl:
	PYTHONPATH=./ python train/run_ner.py simqasl
run_qasl:
	PYTHONPATH=./ python train/run_ner.py qasl
run_mrc:
	PYTHONPATH=./ python train/run_ner.py mrc
