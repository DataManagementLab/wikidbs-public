# WikiDBs: A large-scale corpus of relational databases from wikidata 

This repository contains code to create relational databases based on Wikidata (https://www.wikidata.org/).

# Setup

## Part 1: Setup MongoDB via Docker

The databases will be created based on the Wikidata JSON export. For efficient querying the data will be stored in MongoDB.

1. Make sure that Docker is installed on your system (otherwise install it)
`docker --version`
2. Download the MongoDB docker image from DockerHub
`docker pull mongo`
3. Ensure that the image has been installed
`docker images`
4. Create a `mongo-data` and a `mongo-config` folder to save the data and configuration files in
5. Adapt the `docker-compose.yaml` file found in the `mongodb` folder of this repository to your system (container name, paths to folders, user id)
6. Start MongoDB by running

```
docker-compose up mongo
```

## Part 2: Create a virtual environment and install the requirements.txt

```
python -m virtualenv <env-name>
```

then activate your environment and run:

```
python -m pip install -r requirements.txt
```

## Part 3: Load Wikidata into the MongoDB

We provide two options, the first one (3a) is to import our pre-processed MongoDB export archive file into your MongoDB instance, which will save a lot of time and effort. If you want to do all the necessary steps from scratch yourself, refer to option 3b:

### 3a: Load our pre-processed Wikidata MongoDB export

1. Copy the wikidata_mongodb_archive.gz file from our downloads to the mongo-data folder that you created in step 1.4.

2. Log into the MongoDB Shell by executing:

```
docker exec -it <container_name> bash
```

3. Import the archive by running:

```
 mongorestore --archive=data/db/wikidata_mongodb_archive.gz --gzip --verbose
```

(this should take around 30 minutes)


### 3b: Pre-processing from scratch

To do the pre-processing of the dump from scratch, follow these steps:

1. Download the Wikidata dump
The dataset is based on the Wikidata json dump, the latest dump ("latest-all.json.gz") can be downloaded here: https://dumps.wikimedia.org/wikidatawiki/entities/ (around 115GB)

```
 wget https://dumps.wikimedia.org/wikidatawiki/entities/ 
```

Information page for downloads:
https://www.wikidata.org/wiki/Wikidata:Database_download


2. Preprocess the wikidata dump
To load the dump into MongoDB, adapt the settings in 'conf/preprocess.yaml' and then run the following script:

```
python ./scripts/preprocessing/preprocess_dump.py
```

This will take around 50h.

3. Convert the profiling dictionary into jsonl format
Adapt the settings in 'conf/convert.yaml' and then run the following script:

```
run_exp -m "Wikidata convert profiling dict" -n 0 -- python3 ./scripts/preprocessing/convert_jsonlines.py
```

This will take max. 40h, depending on the settings for 'label_names_min_num_rows'.

# Create Databases

Adapt the settings in 'conf/databases.yaml' to your needs.

We provide a script to run the full pipeline to create database per database, and we provide performance optimized scripts for the stages of crawling, renaming and postprocessing individually.

To run the full pipeline: 

```
python ./scripts/create_databases_full_pipeline.py
```

To run the individual steps:

1. The crawling will create databases from the Wikidata MongoDB dump 
```
python ./scripts/crawl_databases.py
```

2. The renaming will paraphrase table and column names using GPT-4 (adapt conf/rename.yaml)
```
python ./scripts/rename_databases.py
```

3. The postprocessing will transform each database into the final output format (adapt conf/postprocess.yaml)
```
python ./scripts/postprocess.py
```