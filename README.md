# AlgoRhythms

AlgoRhythms is a proof-of-concept Spotify recommendation engine developed as a capstone project by three students at the University of Michigan. It utilizes the Spotify Million Playlists dataset as a basis to inform its knowledge of the relationships between songs.

Description of the dataset used as well as instructions for access are located here: [https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

The project is organized into four subdirectories containing distinct elements of the pipeline. The sections are described in order of the pipeline below.

## Data Engineering

This stage takes the data from the initial dataset to a large adjacency graph where each node is a song and each weighted edge indicates the frequency with which two songs appeared in playlists together. The code is written in Python, utilizing PySpark to handle the large size of the dataset. Because of that it is relatively intensive to initialize the environment and requires that a sufficiently sized cluster is available for use.

There are seven stages to the pipeline that must be executed as separate jobs, in order.

Shell scripts are provided to kick off each job. They correspond directly . They are located [here](./data_engineering/code/scripts)

## Models

The models subdirectory contains all code and notebooks related to the training of machine learning models on the large adjacency graph produced by the data engineering stage.

The embeddings for the 2.2+ million songs in the dataset produced by the graph convolutional neural network can be loaded into an Elasticsearch index using the [es_ingest.py](./es_ingest.py) script. Its requirements are located in the top-level [requirements.txt](./requirements.txt). The command line arguments for the script can be viewed by running `python es_ingest.py --help`.

# How to setup Graph data required for model training

#### Pre-requisites:
- Set up a new file `data_engineering/code/conf/env-secrets.conf` based on `data_engineering/code/conf/env-secrets-template.conf`
- Set up a new file `data_engineering/code/conf/virtual_env.sh` based on `data_engineering/code/conf/virtual_env-template.sh`
- Ensure runtime environment has packages listed in `requirements.txt`
- Once access has been obtained to the dataset, place it in `data` folder within a new directory `spotify_million_playlist_dataset` that is on the same level as your folder path specified in `spark_table_warehouse` variable in `data_engineering/code/conf/virtual_env.sh`. 
It should be structured like below:
![Location of spotify_million_playlist_dataset](https://i.imgur.com/5OhXMZU.png)
#### Instructions for execution:
Since most of the transformations have been achieved through parallelized workloads using the Apache Spark framework, the setup relies on having a cluster to run jobs. For this project, our environment was set up to run using Slurm resource manager. Hence the shell scripts in `data_engineering/code/shell-scripts` have been written to be run in an SBATCH environment.   
**Note:** In case you are other using other cluster environments (E.g. YARN) , make sure to have changes in shell scripts to utilize the respective underlying batch systems .
1. Run `data_engineering/code/shell-scripts/batch-job-1-load-smp.sh` to transform the input.
2. Run `data_engineering/code/2.1_throttled_read_spotifyapi_tracks.py` (i.e. not as shell script to avoid running parallel threads) to gather information of the unique tracks.
3. Run `data_engineering/code/shell-scripts/batch-job-2-gather-unique-track-artist.sh` and `data_engineering/code/shell-scripts/batch-job-2-tracks-csv-to-table.sh` next to populate Spark tables.
4. Run `data_engineering/code/shell-scripts/batch-job-3-compute_track_track_wts.sh` to transform data to weighted relationship data and persist to Spark tables.
5. OPTIONAL : Run `data_engineering/code/shell-scripts/batch-job-4-gather-node-degree-info.sh` and `data_engineering/code/shell-scripts/batch-job-5-generate-sample-data-for-nn.sh` for purposes of  data used in visualizations and sample data for prototype models respectively. 
6. Run `data_engineering/code/shell-scripts/batch-job-5-generate-sample-data-for-nn.sh`to produce the `PyTorch-geometric` Data object that is saved to disk with version number and time stamp ( These files will be referenced by models later for training and embedding generation) 

All the outputs generated, if persisted, can be found in the path defined in `spark_table_warehouse` variable in `data_engineering/code/conf/virtual_env.sh`


# How to train the Graph neural network and generate embeddings
​
#### Pre-requisites:
Before running these steps, it is important to have generated data using the scripts within `data_engineering` folder (described earlier)
- Set up the `datawarehousedir` variable in `models/scripts/conf/models.conf` file to point scripts to location of data generated earlier. It should be the exact same base path used as `spark_table_warehouse` in `data_engineering/code/conf/virtual_env.sh`
- Ensure runtime environment has packages listed in `requirements.txt`
​
#### Instructions for execution:
For this project, our environment was set up to run using Slurm resource manager. Hence the shell scripts in `models/scripts/shell_scripts` have been written to be run in an SBATCH environment.   
**Note:** In case you are other using other cluster environments, make sure to have changes in shell scripts to utilize the respective underlying batch systems .
​
There are 3 main steps to run, in the order listed below:
1. Train the Graph neural network using shell script `models/scripts/shell_scripts/batch_gcnconv.sh` 
2. The above step generates model artifacts, which can be used on the whole data to generate 10-feature embeddings for each song. That is achieved using `models/scripts/shell_scripts/batch_loadembeddings.sh`
3. Using the embeddings generated in step 2 and original graph node features previously available, a feed-forward neural network is trained. This is achieved using `models/scripts/shell_scripts/batch_feedforward.sh`
​
All the outputs generated (embeddings and model artifacts) can be found in `saved_files` folder within the file path specified in `datawarehousedir`

## Backend

The AlgoRhythms backend is a REST API endpoint that creates recommendations based on a list of provided songs. It is implemented as a Lambda function and is thus dependent on being deployed to AWS Lambda. The lambda function requires a layer to be built that contains the dependencies located in the [requirements.txt](./backend/requirements.txt). The required runtime is `Python 3.10`.

The layer can be built as a zip file on Windows using the following commands. The only required dependency to build the layer is Docker. This will produce a zip file containing the dependencies that should be sufficiently sized to use as a Lambda layer.

```
docker run 
```

It also requires several artifacts produced by the Models stage of the pipeline. These artifacts must be hosted in an S3 bucket and the keys pointing to the files must be set as environment variables.

```
One Hot Encoder     -- Used to encode discrete values contained in a song's features
Min Max Scaler      -- Used to scale input features to within the same range initially used to train the model
MLP Weights One     -- Numpy array containing weights for the first layer of the trained MLP
MLP Bias One        -- Numpy array containing biases for the first layer of the trained MLP
MLP Weights Two     -- Numpy array containing weights for the second layer of the trained MLP
MLP Bias Two        -- Numpy array containing biases for the second layer of the trained MLP
```

The function requires the following set of environment variables to be set for the Lambda function:

```
DATABASE_URL    -- URL of an Elasticsearch instance
DATABASE_INDEX  -- The name of index in Elasticsearch
DATABASE_USER   -- Name of Elasticsearch user
DATABASE_PASS   -- Password for Elasticsearch user

MODEL_BUCKET    -- Name of S3 bucket containing artifacts described above
OHE_KEY         -- Key pointing to the One Hot Encoder
SCALER_KEY      -- Key pointing to the Min Max Scaler
WEIGHTS1_KEY    -- Key pointing to the MLP Weights One
WEIGHTS2_KEY    -- Key pointing to the MLP Weights Two
BIAS1_KEY       -- Key pointing to the MLP Bias One
BIAS2_KEY       -- Key pointing to the MLP Bias Two
```

Currently the endpoint is accessible here: `https://ynegl80fpg.execute-api.us-east-1.amazonaws.com/default/algorhythmsAsk`

The endpoint accepts `POST` requests with a JSON body and the schema is as follows:

```json
{
    "count": int,   // Number of recommendations to return
    "songs": [
        {
            "id": string,   // Spotify uri for the track
            "features": {   // Track audio feature vector obtained from this API endpoint: https://developer.spotify.com/documentation/web-api/reference/get-audio-features
                "acousticness": float,
                "danceability": float,
                "duration_ms": int,
                "energy": float,
                "instrumentalness": float,
                "key": int,
                "liveness": float,
                "loudness": float,
                "mode": int,
                "speechiness": float,
                "tempo": float,
                "time_signature": int,
                "valence": float
            }
        },
        ...
    ]
}
```

## Frontend

The frontend is implemented using ReactJS. It is hosted at the following URL: [http://algorhythms-frontend.s3-website-us-east-1.amazonaws.com/](http://algorhythms-frontend.s3-website-us-east-1.amazonaws.com/).

It accesses the AlgoRhythms backend and the Spotify Developer API over HTTP request, and thus is dependent on the AlgoRhythms backend still being hosted and having a valid Spotify account with developer access to the application.

To build and deploy the frontend, you will need NodeJS version `16.0` or greater and to run the following commands at shell or Windows command prompt.

```sh
cd ./frontend
npm install --force
npm run             # This will launch the development server so that the frontend can be observed locally.
npm build           # This will build the application into ./static, where it can then be deployed
```
