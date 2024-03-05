import os

def Generate_Model_Op(model_path, train_data_path):
    

    gt, inp = train_data_path

    model_name = model_path.split('/')[-1].split('.')[0]

    cmd  = f'python scripts/Flare7K/test_large.py --input {inp} --output ./output/{model_name} --model_path {model_path} --flare7kpp'  
    os.system(cmd)
    return


from metrics import Metrics
from glob import glob
import os
import json

def generate_metrics(model_path, gt):
    model_name = model_path.split('/')[-1].split('.')[0]

    gt = train_data_path[0]
    pred = f'./output/{model_name}' + '/blend'

    og_list = glob(gt + '/*')
    pred_list = glob(pred + '/*')

    og_list = sorted(og_list)
    pred_list = sorted(pred_list)

    results_path = f'./metircs_op/{model_name}'
    metrics = Metrics()
    info, df = metrics.run(og_list, pred_list)

    # make a new directory
    os.makedirs(results_path)
    # df.to_csv(f'./results/{unq_tag}/results.csv')   
    df.to_csv(f'{results_path}/results.csv')

    # write the dict to a json file
    # with open(f'./results/{unq_tag}/results.json', 'w') as f:
    with open(f'{results_path}/results.json', 'w') as f:
        json.dump(info, f)



if __name__ == '__main__':
    model_path = '/media/cilab/data/cvpr2024/flare_ensemble/FlareRemoval/model/iitm_dataset.pth'
    train_data_path = ['/home/cilab/teja/FlareRemoval/datasets/input',
                       '/home/cilab/teja/FlareRemoval/datasets/gt']
    
    
    # Generate_Model_Op(model_path, train_data_path)
    generate_metrics(model_path, train_data_path)