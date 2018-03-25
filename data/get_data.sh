#!/usr/bin/env bash

download_data() {
  # Download data from Kaggle
  echo "Downloading data from Kaggle"
  kaggle competitions download --competition donorschoose-application-screening --path raw

  # Unzip and separate data
  echo "Unzipping data"
  unzip raw/train.zip -d input
  unzip raw/test.zip -d input
  unzip raw/resources.zip -d input
  unzip raw/sample_submission.zip -d output
}

# Check that kaggle Python package is installed
if [ $(pip list --format=columns | grep kaggle | wc -l) = 0 ];
then
  echo "Kaggle package not found"
else
  download_data
fi
