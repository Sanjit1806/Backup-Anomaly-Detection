import pandas as pd
import random  as r


# Basic features involved in the data
cols = ['total_size', 'no_of_files', 'network_throughput', 'time_taken']
data = list()

# Generation of the random data
for _ in range(1000):
    # Generate random numbers within a specific range
    size, nt = r.randint(6,195), r.randint(3,98)
    nf = r.randint(1,150)
    time_approx = size/nt if nf<=20 else (1.2 * size/nt if nf<=50 else (1.4 * size/nt if nf<=100 else 1.6 * size/nt))
    # Append the random data generated to the list 'data'
    data.append([
    r.randint(int((size - 5)*10), int((size + 5)*10)) * 0.1,
    r.randint(nf, nf + 25),
    r.randint(int((nt-2)*100), int((nt+2)*100)) * 0.01,
    r.randint(int((time_approx)*10), int((time_approx+5)*10)) * 0.1
    ])

# Create a pandas DataFrame.
df = pd.DataFrame(data, columns=cols)
new_data = pd.DataFrame(columns=cols+['size_to_nt'])

# Add a feature which indicates the ratio of the total size to the network throughput
df['size_to_nt'] = df.total_size / df.network_throughput


def fetch_train_data():
    global df, new_data
    df = pd.concat([df.loc[new_data.shape[0]:], new_data])
    new_data = pd.DataFrame(columns=cols+['size_to_nt'])
    return df

def fetch_new_data(qty=1):
    global new_data, df
    data = list()
    for _ in range(qty):
        size, nt, nf = r.randint(6,195), r.randint(3,98), r.randint(1,150)
        time_approx = size/nt if nf<=20 else (1.2 * size/nt if nf<=50 else (1.4 * size/nt if nf<=100 else 1.6 * size/nt))
        total_size = r.randint(int((size - 5)*10), int((size + 5)*10)) * 0.1
        net_thru = r.randint(int((nt-2)*100), int((nt+2)*100)) * 0.01
        data.append([
        total_size,
        r.randint(nf, nf + 25),
        net_thru,
        r.randint(int((time_approx)*10), int((time_approx+5)*10)) * 0.1,
        total_size/net_thru
        ])

    new_data_point = pd.DataFrame(data=data, columns=cols+['size_to_nt'])
    new_data = pd.concat([new_data, new_data_point])
    return new_data_point
