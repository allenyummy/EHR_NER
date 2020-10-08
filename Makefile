test:
	PYTHONPATH=./ pytest --log-cli-level=debug --cov=train/ --cov=utils/datasets/ --cov=configs/ --cov=tests/
freeze:
	poetry export --without-hashes -f requirements.txt --output requirements.txt
build:
	docker build . -t ehr_ner:0.1.0
