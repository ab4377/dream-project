import pandas as pd
import numpy as np
import dream

def main():
    df = pd.read_csv('../data/23100f3e-bbcb-4fc1-b8e3-18c94a611e38/deviceMotion_walking_return.json.items.csv')
    print(dream.find_forward_backward_direction(df, np.pi / 100, visualize=True, visualize_path='../data/plots'))

if __name__ == '__main__':
    main()