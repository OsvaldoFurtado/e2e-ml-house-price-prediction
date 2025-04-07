download-dataset:
	acquire-dataset -o artefacts/raw_dataset -k ~/Downloads/kaggle.json -z carlmcbrideellis/data-anscombes-quartet
	@echo "Dataset downloaded to artefacts/raw_dataset"