import os
from google.cloud import storage
from google.cloud import bigquery


#the bigquery class is use to submit a query, and then save it as a temp table under a temp data set.
#retrive the table from bigquery to pandas dataframe (local).
class BigQueryService:
    def __init__(self, service_key_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_path
        self.client = bigquery.Client()

    def query_to_table (self, query, table_id): #query to big query table
        job_config = bigquery.QueryJobConfig(destination = table_id)
        query_job = self.client.query(query, job_config)
        query_job.result()
        print("Query results loaded to the table {}".format(table_id))

    def query_to_df(self,query): #query to pandas dataset local
        df = self.client.query(query).to_dataframe()
        return(df)

    def table_to_df(self,project_name, bq_dataset_name, bq_table_name):
        dataset_ref = self.client.dataset(bq_dataset_name, project = project_name)
        table_ref = dataset_ref.table(bq_table_name)
        table = self.client.get_table(table_ref)
        df = self.client.list_rows(table).to_dataframe()
        return(df)