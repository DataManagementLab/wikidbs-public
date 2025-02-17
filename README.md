![WikiDBs logo](assets/WikiDBs.png)
![WikiDBs authors](assets/authors.png)

# WikiDBs: A large-scale corpus of relational databases from wikidata 

This repository contains the code for WikiDBs (https://wikidbs.github.io/), a corpus of relational databases based on Wikidata (https://www.wikidata.org/).

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

as well as

```
python -m pip install --editable .
```

## Part 3: Load Wikidata into the MongoDB

We provide two options, the first one (3a) is to import our pre-processed MongoDB export archive file into your MongoDB instance, which will save a lot of time and effort. If you want to do all the necessary steps from scratch yourself, refer to option 3b:

### 3a: Load our pre-processed Wikidata MongoDB export

1. Copy the wikidata_mongodb_archive.gz (~ 13.2GB) file from our [downloads](https://drive.google.com/drive/folders/1wMRFro0ydQghmYeavBaBv_IUsPobT_JK?usp=sharing) to the mongo-data folder that you created in step 1.4. Around 40GB of disk space are required for the mongo-data folder. 

2. Log into the MongoDB Shell by executing:

```
docker exec -it <container_name> bash
```

3. Import the archive by running:

```
 mongorestore --archive=data/db/wikidata_mongodb_archive.gz --gzip --verbose
```

(this takes around 60 minutes on an Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz)


### 3b: Pre-processing from scratch

To do the pre-processing of the dump from scratch, follow these steps:

1. Download the Wikidata dump
The dataset is based on the Wikidata json dump, the latest dump ("latest-all.json.gz") can be downloaded here: https://dumps.wikimedia.org/wikidatawiki/entities/ (needs around 115GB of disk space)

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

Adapt the settings in our config files, especially 'conf/databases.yaml' to your needs.

We provide performance optimized scripts for the stages of crawling, renaming and postprocessing individually. On average around 5MB of disk space are necessary for each created database.

Our scripts are scalable and the number of workers for creating databases in parallel can be specified in our configuration file.
On an Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz we observe the following resource consumption:
1 CPU core per worker and ~25GiB RAM per worker.
Each worker creates approximately 20 databases per hour on our system.

You'll need the "wikidata_labels.json" and the "wikidata_properties.json" files found in our [downloads](https://drive.google.com/drive/folders/1wMRFro0ydQghmYeavBaBv_IUsPobT_JK?usp=sharing).

To run the pipeline:

1. The crawling will create databases from the Wikidata MongoDB dump 
```
python ./scripts/crawl_databases.py
```

2. The renaming will paraphrase table and column names using the OpenAI API with batch processing (adapt conf/rename.yaml)
```
python ./scripts/rename_databases.py
```

3. The postprocessing will transform each database into the final output format (adapt conf/postprocess.yaml)
```
python ./scripts/postprocess.py
```

4. The finalize script will bring the databases in the exact for format used for the WikiDBs corpus, with the option to split them into multiple subfiles (adapt settings in scripts/finalize.py)
```
python ./scripts/finalize.py
```
