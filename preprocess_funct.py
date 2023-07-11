import glob
import warnings
import numpy as np

data_shuffle_epoch = 3  
total_size = 1  
train_size = 0.8  

AEDGAN_folder_path = './Jupyter/AEDGAN/'
processed_data_path = AEDGAN_folder_path + 'Processed_data/'
csv_data_path = processed_data_path

def preprocess_data1(seq_len,target_len):
    
    input_len = seq_len 
    output_len = target_len 
    seq_len_total = input_len + output_len

    data_path1 = csv_data_path + str(output_len) + "sequence/" + str(input_len) + "sequence/"

    num_csv = 0
    for csv_file in glob.glob(data_path1 + '*.csv'):
        num_csv += 1
    # print("num_csv:", num_csv) 


    i = 1

    print("Reading csv file...")
    while i <= num_csv:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xy_temp = np.loadtxt(data_path1 + str(i) + ".csv",delimiter=",", dtype=np.str)
        if len(xy_temp) != 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xy_before_process = np.loadtxt(data_path1 + str(i) + ".csv",delimiter=",", dtype=np.str)
            z = i+1
            i = num_csv + 1
        i += 1
  
    for j in range(z, num_csv+1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xy_temp = np.loadtxt(data_path1 + str(j) + ".csv",delimiter=",", dtype=np.str)
        if len(xy_temp) != 0:
            xy_before_process = np.vstack((xy_before_process, xy_temp))

    # print("xy_before_process", np.shape(xy_before_process))  
    print("Done reading csv file.")


    print("Processing data...")
    # process the data from the csv file
    processed_data = []
    for rows in xy_before_process:
        temp_processed_data_1 = []
        for i in range((seq_len_total + 0)):
            temp_processed_data_2 = []
            for j in range(
                int((len(rows) / (seq_len_total + 0)) * i),
                int((len(rows) / (seq_len_total + 0)) * (i + 1)),
            ):
                rows[j] = rows[j].replace("[", "")
                rows[j] = rows[j].replace("]", "")
                rows[j] = rows[j].replace('"', "")
                temp_processed_data_2.append(float(rows[j]))
            temp_processed_data_1.append(temp_processed_data_2)
        processed_data.append(temp_processed_data_1)
    # print(processed_data)
    # print(np.shape(processed_data))
    # print(type(processed_data))

    processed_data2 = np.array(processed_data)

    ### randomize the data
    for data_shuffle in range(data_shuffle_epoch):
        #total_data = shuffle(processed_data, random_state=0)
        np.random.shuffle(processed_data2)
    total_data = processed_data2

    print("Done processing data.")
    print(np.shape(total_data))
    print(type(total_data))



    ### decide how much data to be used. If total_size is  1, all data will be used
    total_data_length = int(total_size * len(total_data))
    total_data = total_data[0:total_data_length]

    # Spliting training and testing data.  
    # If train_size is 0.7, 70% training and 30% test 
    # No validation data
    train_data_length = int(train_size * len(total_data))
    train_data = total_data[0:train_data_length]
    test_data = total_data[train_data_length:-1]


    train_data = np.expand_dims(train_data, axis=3)
    test_data = np.expand_dims(test_data, axis=3)


    print("train_data_shape: ", np.shape(train_data))
    print("test_data_shape: ", np.shape(test_data))

    
    return train_data, test_data