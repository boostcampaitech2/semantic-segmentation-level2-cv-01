import pandas as pd
import tqdm

# CSV path.
df752 = pd.read_csv('/opt/ml/segmentation/mmsegmentation/Second/752_hard.csv')
df727 = pd.read_csv('/opt/ml/segmentation/mmsegmentation/Second/727_DPT.csv')
df725 = pd.read_csv('/opt/ml/segmentation/mmsegmentation/Second/725_setr.csv')
df716 = pd.read_csv('/opt/ml/segmentation/mmsegmentation/Second/716_DPT.csv')
df704 = pd.read_csv('/opt/ml/segmentation/mmsegmentation/Second/704_uper.csv')
df684 = pd.read_csv('/opt/ml/segmentation/mmsegmentation/Second/684_effi.csv')

# Make ensemble empty DataFrame
ensemble_df = pd.DataFrame(index=range(0,819), columns=['image_id','PredictionString'])

# ensemble
mx = 0
for i in tqdm.tqdm(range(819)):
    '''
    mx - most label index variable.
    vote - temp vote label list
    temp_pred - temp image prediction string list
    cnt - temp, label in vote count list

    '''
    df752_pred = df752['PredictionString'][i].split()
    df727_pred = df727['PredictionString'][i].split()
    df725_pred = df725['PredictionString'][i].split()
    df716_pred = df716['PredictionString'][i].split()
    df704_pred = df704['PredictionString'][i].split()
    df684_pred = df684['PredictionString'][i].split()
    temp_pred = []
    for j in range(65536):
        vote = []
        vote.append(df752_pred[j])
        vote.append(df727_pred[j])
        vote.append(df725_pred[j])
        vote.append(df716_pred[j])
        vote.append(df704_pred[j])
        vote.append(df684_pred[j])      
        cnt = []
        # range(num) ; num - number of csv files
        for k in range(6):
            cnt.append(vote.count(vote[k]))
        mx = cnt.index(max(cnt))
        temp_pred.append(vote[mx])
    ensemble_df['image_id'][i] = df727['image_id'][i]
    ensemble_df['PredictionString'][i] = ' '.join(str(e) for e in temp_pred)

# save csv file.
ensemble_df.to_csv('/opt/ml/segmentation/mmsegmentation/Second/hard_752_727_725_716_704_684.csv', index=False)
        