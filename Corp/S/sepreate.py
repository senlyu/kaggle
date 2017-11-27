import pandas as pd
import numpy as np
import os



data = pd.read_csv('../../../data/train.csv')
new = np.array_split(data,3)
output_path = os.path.join('~/git/data','train1.csv')
new[0].to_csv(output_path,index=False)
output_path = os.path.join('~/git/data','train2.csv')
new[1].to_csv(output_path,index=False)
output_path = os.path.join('~/git/data','train3.csv')
new[2].to_csv(output_path,index=False)