import pandas as pd
import numpy as np
import subprocess, re, os, time

from multiprocessing import Pool


from tqdm import tqdm

all_eval_releases = ['activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
                     'camel-2.10.0', 'camel-2.11.0',
                     'derby-10.5.1.1',
                     'groovy-1_6_BETA_2',
                     'hbase-0.95.2',
                     'hive-0.12.0',
                     'jruby-1.5.0', 'jruby-1.7.0.preview1',
                     'lucene-3.0.0', 'lucene-3.1',
                     'wicket-1.5.3']

all_dataset_name = ['activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket']

base_file_dir = '../../../datasets/ErrorProne_data/'
base_command = "javac -J-Xbootclasspath/a:javac-9+181-r4173-1.jar -XDcompilePolicy=simple -processorpath error_prone_core-2.4.0-with-dependencies.jar;dataflow-shaded-3.1.2.jar;jFormatString-3.0.0.jar  \"-Xplugin:ErrorProne -XepDisableAllChecks -Xep:CollectionIncompatibleType:ERROR\" "

result_dir = './ErrorProne_result/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def run_ErrorProne(rel):
    df_list = []
    java_file_dir = base_file_dir+rel+'/'

    file_list = os.listdir(java_file_dir)

    for java_filename in tqdm(file_list):
        f = open(java_file_dir+java_filename,'r',encoding='utf-8',errors='ignore')
        java_code = f.readlines()

        code_len = len(java_code)

        output = subprocess.getoutput(base_command+java_file_dir+java_filename)
        print(output)
        reported_lines = re.findall('\d+: error:',output)
        reported_lines = [int(l.replace(':','').replace('error','')) for l in reported_lines]
        reported_lines = list(set(reported_lines))
        line_df = pd.DataFrame()
        line_df['filename'] = [java_filename.replace('_','/')]*code_len
        line_df['test-release'] = [rel]*len(line_df)
        line_df['line_number'] = np.arange(1,code_len+1)
        line_df['EP_prediction_result'] = line_df['line_number'].isin(reported_lines)

        df_list.append(line_df)

    final_df = pd.concat(df_list)
    final_df.to_csv(result_dir+rel+'-line-lvl-result.csv',index=False)
    print('finished',rel)

agents = 5
chunksize = 8

# with Pool(processes=agents) as pool:
#     pool.map(run_ErrorProne, all_eval_releases, chunksize)
for rel in all_eval_releases:
    run_ErrorProne(rel)
