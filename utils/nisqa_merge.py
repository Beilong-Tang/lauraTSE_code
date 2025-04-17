import pandas as pd
import argparse 

p = argparse.ArgumentParser()
p.add_argument('--output_dir', required=True, type = str)
args = p.parse_args()


file_path = f'{args.output_dir}/NISQA_results.csv'
file = pd.read_csv(file_path)
context = file['mos_pred'].mean()
context = 'NISQA average mos: ' + str(context)
print(context)
with open(f'{args.output_dir}/NISQA_mos.txt', 'w') as f:
    print(context, file=f)