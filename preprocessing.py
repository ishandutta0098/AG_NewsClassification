#File Path
TRAIN_FILE_PATH = '/content/MyDrive/MyDrive/Data Science/Projects/SpoonShot/train.csv'
TEST_FILE_PATH = '/content/MyDrive/MyDrive/Data Science/Projects/SpoonShot/test.csv'

#Load Data
data = pd.read_csv(TRAIN_FILE_PATH)
testdata = pd.read_csv(TEST_FILE_PATH)

#Set Column Names 
data.columns = ['ClassIndex', 'Title', 'Description']
testdata.columns = ['ClassIndex', 'Title', 'Description']

#Combine Title and Description
X_train = data['Title'] + " " + data['Description'] # Combine title and description (better accuracy than using them as separate features)
y_train = data['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0

x_test = testdata['Title'] + " " + testdata['Description'] # Combine title and description (better accuracy than using them as separate features)
y_test = testdata['ClassIndex'].apply(lambda x: x-1).values # Class labels need to begin from 0

#Max Length of sentences in Train Dataset
maxlen = X_train.map(lambda x: len(x.split())).max()
data.head()
