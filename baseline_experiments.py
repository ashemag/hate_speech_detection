"""
Runs logistic regression
"""
import configparser
from data_providers import LogisticRegressionDataProvider

config = configparser.ConfigParser()
config.read('config.ini')


if __name__ == "__main__":
    path_data = config['DEFAULT']['PATH_DATA']
    path_labels = config['DEFAULT']['PATH_LABELS']
    data = LogisticRegressionDataProvider(path_data, path_labels).extract()
    print(data.keys())
    # data = extract_data(args.embedding, args.embedding_level, args.model)
    # x_train, y_train, x_val, y_val, x_test, y_test = LogisticRegressionDataProvider().extract(FILENAME, FILENAME_LABELS)