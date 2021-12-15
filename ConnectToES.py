import os
import time
import warnings
import json
from elasticsearch import Elasticsearch

# ignore the warnings which is rose cause we are using an older version of Elasticsearch
# or the hard disk saving the data is about to get full.
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


# connecting the Elasticsearch index
def connectToES():
    # return the paths for the programs and the flag for auto starting the programs
    programs_info = json.load(open('Objects/programs_info.json'))

    # return the host, port and index for the Elasticsearch
    es_params = json.load(open('Objects/es_parameters.json'))

    if programs_info["auto_start"] == "True":
        # starting the Elasticsearch using command line on windows
        # you have to change the paths for Elasticsearch and Kibana when using the system
        # on a different device or you can simply start the programs before starting the system
        print("Starting Elasticsearch and Kibana ...")
        os.startfile(programs_info['elastic_path'])
        os.startfile(programs_info['kibana_path'])
        time.sleep(120)
        print("Done.")

    while True:
        try:
            # connecting to Elasticsearch to pass the tweets into an index
            es = Elasticsearch([{'host': es_params['host'], 'port': es_params['port']}])
            # creating an index
            es.indices.create(index=es_params['index'], ignore=400)
            print("Kibna and Elasticsearch index is ready.")
            return es

        # which occur when the Elasticsearch faces an error
        except:
            print("couldn't connect to ElasticSearch ... waiting 5 second.")
            time.sleep(5)
            continue


