import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.nn.functional import dropout
from torch.serialization import save
from others.logging import logger, init_logger

class Preprocessing:
    def __init__(self, data_path, save_path, shuffle=False, random_state=21, train_test_split=False, tag=''):
        
        # init_logger()
        
        self.data_path = data_path
        self.save_path = save_path
        self.tag = tag
        self.shuffle = shuffle
        self.random_state = random_state
        self.train_test_split = train_test_split


    def preprocess(self):
        if not self.data_path:
            raise Exception('Data path not set.')
        else:
            data = pd.read_csv(self.data_path)[['title', 'abstract']]
            if self.shuffle:
                logger.info('Shuffling data.')
                data = shuffle(data, random_state=self.random_state).reset_index(drop=True)

            X = data['abstract']
            y = data[['title']]
            

            if self.train_test_split:
                logger.info('Splitting data.')
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=self.random_state+1)
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=15/85, random_state=self.random_state+2)
                for d in (X_train, X_val, y_train, y_val, X_test, y_test):
                    d = d.reset_index(drop=True)

                logger.info('Train test split complete.')
                logger.info(f'Training size: {X_train.shape}')
                logger.info(f'Validation size: {X_val.shape}')
                logger.info(f'Test size: {X_test.shape}')
                logger.info(f'Saving data at {self.save_path}')
                X_train.to_csv(f'{self.save_path}/X_train{self.tag}.txt', header=None, index=None, sep='\n')
                y_train.to_csv(f'{self.save_path}/y_train{self.tag}.txt', header=None, index=None, sep='\n')
                X_val.to_csv(f'{self.save_path}/X_val{self.tag}.txt', header=None, index=None, sep='\n')
                y_val.to_csv(f'{self.save_path}/y_val{self.tag}.txt', header=None, index=None, sep='\n')
                X_test.to_csv(f'{self.save_path}/X_test{self.tag}.txt', header=None, index=None, sep='\n')
                y_test.to_csv(f'{self.save_path}/y_test{self.tag}.txt', header=None, index=None, sep='\n')



        logger.info('Preprocessing complete.')
                
                
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", default='../data/')
    parser.add_argument("-save_path", default='../data/')

    # parser.add_argument("-shard_size", default=2000, type=int)
    # parser.add_argument('-min_nsents', default=3, type=int)
    # parser.add_argument('-max_nsents', default=100, type=int)
    # parser.add_argument('-min_src_ntokens', default=5, type=int)
    # parser.add_argument('-max_src_ntokens', default=200, type=int)

    # parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)

    parser.add_argument('-log_file', default='../logs/my_model.log')


    # parser.add_argument('-n_cpus', default=2, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    preprocess = Preprocessing(data_path='..\data\papers_small.csv', save_path='..\data', shuffle=True, train_test_split=True, tag='_small')
    preprocess.preprocess()