# Data

The data for this project comes from the [DonorsChoose.org Application Screening](https://www.kaggle.com/c/donorschoose-application-screening/data) competition.

## Getting Started

To download and unzip the data, run the following command in your `bash` shell in the `data` directory.

```bash
./get_data.sh
```

This will check that the Kaggle Python package is installed, download the data, unzip it, and put it in the appropriate directories.

## Directories

Once the `get_data.sh` command has been run, several directories will be created:

* `raw`:  This directory contains the raw `zip` files downloaded from Kaggle.
* `input`:  This directory contains the input data files: `train.csv`, `test.csv`, and `resources.csv`
* `output`:  This directory contains the output file `sample_submission.csv`.
