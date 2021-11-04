# semantic-segmentation-level2-cv-01
semantic-segmentation-level2-cv-01 created by GitHub Classroom

### CRF_csv.py
  - Just You need csv!
  ``` python
  df = pd.read_csv('Path/File_name.csv')
  ```
  - Name the path and file you want.
  ``` python
  df2.to_csv('Path/File_name2.csv', index=False)
  ```
  
### ensemble_hard.py
  - Just You need csv! Ensemble file that you want!
  ``` python
  df1 = pd.read_csv('Path/File_name1.csv')
  df2 = pd.read_csv('Path/File_name2.csv')
  df2 = pd.read_csv('Path/File_name3.csv')
  ```
  - Attention! Think about your number of csv file.
  ``` python
  for k in range(num):
    cnt.append(vote.count(vote[k]))
  ```
  - Name the path and file you want. 
  ``` python
  ensemble_df.to_csv('Path/File_name.csv', index=False)
  ```
  
