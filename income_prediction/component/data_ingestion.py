from income_prediction.logger import logging
from income_prediction.exception import IncomeException
from income_prediction.entity.entity_config import DataIngestionConfig
import sys,os
import pandas as pd
from income_prediction.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
import numpy as np

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise IncomeException(e,sys) from e
    def download_url_data(self)->str:
        try:
            download_url=self.data_ingestion_config.dataset_url
            raw_data_dir=self.data_ingestion_config.raw_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir,exist_ok=True)
            file_path=self.data_ingestion_config.raw_data_file
            df=pd.read_csv(download_url)
            df.columns=['age','workclass','fnlwgt','education','education_num','marital_status',
                        'occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week',
                        'native_country','wages']
            df.drop(columns=df.columns[3],axis=1,inplace=True)
            df[df==' ?']=np.nan
            cat=[i for i in df.columns if df[i].dtypes=='O']
            for j in cat:
                df[j]=df[j].str.replace(" ","")
            df.to_csv(file_path,index=False)
            logging.info(f'download data from {download_url} to {file_path}')
            return file_path
        except Exception as e:
            raise IncomeException(e,sys) from e
        
    def split_data_as_train_test(self)->DataIngestionArtifact:
        try:
            adult_income_data=pd.read_csv(self.data_ingestion_config.raw_data_file)
            file=os.path.basename(self.data_ingestion_config.raw_data_file)
            train_set=None
            test_set=None
            train_set,test_set=train_test_split(adult_income_data,test_size=0.25,stratify=adult_income_data.iloc[:,-1],random_state=42)
            train_file_path=os.path.join(self.data_ingestion_config.ingested_train_dir,file)
            test_file_path=os.path.join(self.data_ingestion_config.ingested_test_dir,file)
            
            if train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f'Exporting training data to file [{train_file_path}]')
                train_set.to_csv(train_file_path,index=False)
            if test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,exist_ok=True)
                logging.info(f"Exporting testing dataset to file :[{test_file_path}]")
                test_set.to_csv(test_file_path,index=False)
            data_ingestion_artifact=DataIngestionArtifact(train_file_path=train_file_path,
                                                        test_file_path=test_file_path,
                                                        is_ingested=True,
                                                        message=f'Data ingestion completed successfully')
            logging.info(f'Data ingestion artifact: [{data_ingestion_artifact}]')
            return data_ingestion_artifact        
        except Exception as e:
            raise IncomeException(e,sys) from e 
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_file_path=self.download_url_data()  
            return self.split_data_as_train_test()
        except Exception as e:
            raise IncomeException(e,sys) from e