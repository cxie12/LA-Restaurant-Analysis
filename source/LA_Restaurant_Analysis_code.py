#!/usr/bin/env python
# coding: utf-8

# # Los Anageles Restaurant and Market Health Analysis

# Cheng Xie

# In[1]:


#pip install pandas 
#pip install matplotlib
#pip install h2o
import  warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, classification_report

import h2o
from h2o.frame import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator


# ## Data Loading

# Inspection records

# In[12]:


df_inspect = pd.read_csv('/Users/zhangdoudou/Desktop/BD_Project/restaurant-and-market-health-inspections.csv', parse_dates=['activity_date'])
df_inspect.head()


# In[7]:


print(df_inspect.describe())


# violation records

# In[13]:


df_violate = pd.read_csv('/Users/zhangdoudou/Desktop/BD_Project/restaurant-and-market-health-violations.csv', parse_dates=['activity_date'])
df_violate.head()


# In[5]:


print(df_violate.describe())


# ## Data cleaning and preprocessing 

# 1. violation code: remove codes starting with 'W'
# 2. apply regular expression to the column 'pe_description'
# 3. remove zipcode extensions

# We notice that the violation codes yield codes that mostly begin with a 'F'. However, there seem to be a few starting with 'W' that only appear once or twice. The violation codes starting with 'W' are outliers since there is no matching violation discription with these codes. Considering they only make up 17 entries of the 272,801 violations, we can safely drop them.

# In[6]:


df_violate['violation_code'].value_counts().nsmallest(20)


# In[9]:


df_violate['violation_description'].value_counts()


# In[5]:


df_violate = df_violate[~df_violate['violation_code'].isin(df_violate['violation_code'].value_counts()
                                                           .nsmallest(14).index)]
df_violate.shape


# No discription for F056, thus we can drop the smallest 14 violation codes including all codes starting with 'W' and code F056. 

# Regular Expression

# The column 'pe_description'describes 3 different things: what type of establishment the facility is, how many people it can host, and the risk level. 
# 
# In this part, we would like to use regular expression to define a search pattern that split pe_description comlumn into three parts. To better understand the data, we will then assign it to three new feature variables(facility category, seat capacity and risk level) for further analysis later. 

# In[5]:


df_violate['pe_description'].unique() #many punctuation marks inside


# Define search patterns by functions

# In[6]:


import re
def find_est(str):
    est = re.compile('.+(?= \()')
    return est.search(str).group(0)
def find_size(str):
    size = re.compile('(?<=\().+(?=\))')
    return size.search(str).group(0)
def find_risk(str):
    return (' ').join(str.split(' ')[-2:])


# In[8]:


testre = 'FOOD MKT RETAIL (2,000+ SF) MODERATE RISK'


# In[9]:


print(find_est(testre))
print(find_size(testre))
print(find_risk(testre))


# In[14]:


df_violate['fac_type'] = df_violate['pe_description'].apply(find_est)
df_violate['seats'] = df_violate['pe_description'].apply(find_size)
df_violate['risk'] = df_violate['pe_description'].apply(find_risk)
df_violate.head()


# In[15]:


#apply to inspection datastet 
df_inspect['fac_type'] = df_inspect['pe_description'].apply(find_est)
df_inspect['seats'] = df_inspect['pe_description'].apply(find_size)
df_inspect['risk'] = df_inspect['pe_description'].apply(find_risk)
df_inspect.head()


#  visulize the missing values

# In[11]:


import missingno as msno
msno.matrix(df_violate)


# In[18]:


import missingno as msno
msno.matrix(df_inspect) #visulizing the missing value: 


# zipcode extensions
# When looking at the zipcode, we noticed that some of the zipcode records have extensions for better mailing purpose. To have a deeper understanding of the data,we only take the first five characters. 

# In[16]:


def clean_zip(x):
    return x[:5]
df_inspect['facility_zip'] = df_inspect['facility_zip'].apply(clean_zip)
df_violate['facility_zip'] = df_violate['facility_zip'].apply(clean_zip)
#df_inspect['facility_zip'].head()
#df_violate['facility_zip'].head()


# Reshape dataframe

# Step1: group a new data frame by facility and violation code, then aggregate by the count to find the total times each facility has violated a particular rule. 

# In[17]:


vio_grouped = df_violate.groupby(['facility_id','violation_code'])
vio_counts = vio_grouped.agg({
    'violation_code': 'count'
})
vio_counts.head()


# step2: transpose violation codes into columns 

# In[18]:


vio_matrix = vio_counts['violation_code'].unstack(0).T #rows to columns
vio_matrix.fillna(0, inplace=True) 
vio_matrix.reset_index(inplace=True)
vio_matrix.head()


# In[19]:


import pandas as pd
inspect_vio_df = pd.merge(df_inspect, vio_matrix, on='facility_id', how='left') #merge 
inspect_vio_df.fillna(0, inplace = True)
inspect_vio_df['facility_zip'] = inspect_vio_df['facility_zip'].apply(clean_zip)
inspect_vio_df.head()


# ## Correlation Analysis

# In[23]:


plt.figure(figsize=(10,5))
sns.heatmap(vio_matrix.corr())


# In[24]:


plt.figure(figsize=(10,5))
sns.heatmap(inspect_vio_df.corr())


# In[25]:


corr = vio_matrix.corr()
mask = (corr > 0.5) & (corr != 1.0)
masked = corr[mask]
masked.dropna(how='all', axis = 0, inplace=True)
masked.dropna(how='all', axis = 1, inplace=True)
plt.figure(figsize=(10,5))
sns.heatmap(masked)


# Matching violation codes with violation descriptions

# In[20]:


vio_codes = list(df_violate['violation_code'].unique())
vio_descs = list(df_violate['violation_description'].unique())


# In[41]:


violation_dict = dict.fromkeys(vio_codes) #matching
for x in violation_dict:
    for desc in vio_descs:
        if x[-2:] == desc[2:4]:
            violation_dict[x] = desc
violation_dict


# ## Georgraphical distribution

# In[21]:


geo_group = inspect_vio_df.groupby('facility_zip') #grouping and aggregating
geo_agg = geo_group.agg({'score': np.mean,
                       'facility_id': pd.Series.nunique,
                        })
geo_agg.rename(columns = {'score':'average_score','facility_id':'total_facilities'}, inplace = True)
geo_agg.reset_index(inplace=True)


# In[22]:


geo_agg


# Find data that represents the boundary of each zipcode area
# Data source: http://boundaries.latimes.com/set/zip-code-tabulation-areas-2012/

# In[24]:


# load GeoJSON
import simplejson as json
with open('/Users/zhangdoudou/Desktop/BD_Project/la-zip-code-areas-2012.geojson', 'r') as jsonFile:
    data = json.load(jsonFile)
tmp = data

# remove ZIP codes not in our dataset
geozips = []
for i in range(len(tmp['features'])):
    if tmp['features'][i]['properties']['name'] in list(geo_agg['facility_zip'].unique()):
        geozips.append(tmp['features'][i])

# creating new JSON object
new_json = dict.fromkeys(['type','features'])
new_json['type'] = 'FeatureCollection'
new_json['features'] = geozips

# save JSON object as updated-file
open("updated-file.json", "w").write(
    json.dumps(new_json, sort_keys=True, indent=4, separators=(',', ': '))
)


# In[45]:


import folium

# table = main table/data frame we read from (pandas DataFrame)
# zips = column name where ZIP codes are (string)
# mapped_feature = column name for feature we want to visualize (string)
# add_text = any additional commentary to be added in the map legend (string)

def create_map(table, zips, mapped_feature, add_text = ''):
    # reading of the updated GeoJSON file
    la_geo = r'updated-file.json'
    # initiating a Folium map with LA's longitude and latitude
    m = folium.Map(location = [34.0522, -118.2437], zoom_start = 11)
    # creating a choropleth map
    m.choropleth(
        geo_data = la_geo,
        fill_opacity = 0.7,
        line_opacity = 0.2,
        data = table,
        # refers to which key within the GeoJSON to map the ZIP code to
        key_on = 'feature.properties.name',
        # first element contains location information, second element contains feature of interest
        columns = [zips, mapped_feature],
        fill_color = 'RdYlGn',
        legend_name = (' ').join(mapped_feature.split('_')).title() + ' ' + add_text + ' Across LA'
    )
    folium.LayerControl().add_to(m)
    # save map with filename based on the feature of interest
    m.save(outfile = mapped_feature + '_map.html')


# Score distribution

# In[32]:


create_map(geo_agg, 'facility_zip', 'average_score', 'of Facilities')


# In[ ]:


Facility ditribution


# In[33]:


create_map(geo_agg, 'facility_zip', 'total_facilities')


# violation distribution

# In[46]:


fac_grouped = inspect_vio_df.groupby(['facility_zip','facility_id'])
fac_agg = fac_grouped.agg({'F044': np.mean,
                        'F033': np.mean,
                        'F035': np.mean,
                        'F036': np.mean,
                        'F040': np.mean,
                        'F043': np.mean,
                        'F037': np.mean,
                        'F039': np.mean,
                        'F030': np.mean,
                        'F006': np.mean,
                        'F014': np.mean,
                        'F007': np.mean})


# In[47]:


fac_agg = fac_agg.groupby(level=0).mean()
fac_agg.reset_index(inplace=True)


# In[48]:


create_map(fac_agg, 'facility_zip', 'F044', 'Average Violation of Floor, Walls, and Ceiling Maintenance and Condition')
create_map(fac_agg, 'facility_zip', 'F033', 'Average Violation of Nonfood Equipment Cleanliness and Maintenance')
create_map(fac_agg, 'facility_zip', 'F035', 'Average Violation of Equiptment/Utensils Cleanliness and Maintenance')
create_map(fac_agg, 'facility_zip', 'F036', 'Average Violation of Equiptment/Utensils/Linens Storage and Use')
create_map(fac_agg, 'facility_zip', 'F040', 'Average Violation of Plumbing')
create_map(fac_agg, 'facility_zip', 'F014', 'Average Violation of Surface Cleaning and Sanitization')


# In[25]:


inspect_vio_df.dtypes


# ## Modeling
# 

# objective: grade
# features: risk level, facility type, service code, month, day

# ### Feature Engineering

# Split date to year, month and the day of the week(Mon,Tue,Wed...)

# In[26]:


df_inspect2 = pd.read_csv('/Users/zhangdoudou/Desktop/BD_Project/restaurant-and-market-health-inspections.csv')


# In[27]:


def clean_date(x):
    return x[:10]

date = df_inspect2['activity_date'].astype('str').apply(clean_date)


# In[28]:


inspect_vio_df['month'] = pd.to_datetime(date).dt.month #find month 
inspect_vio_df['day'] = pd.to_datetime(date).dt.dayofweek #find day of the week
inspect_vio_df['year'] = pd.to_datetime(date).dt.year #year


# In[52]:


inspect_vio_df.dtypes


# In[53]:


code_count = df_violate.violation_code.value_counts()
code_count


# Violation code: larger than 9800 or not

# In[32]:


def is_large(x):
   if x == 'F044' or x == 'F033' or x == 'F035' or x == 'F036' or x == 'F040' or x == 'F043' or x == 'F037' or x == 'F039' or x == 'F030' or x == 'F006' or x == 'F004' or x == 'F014' or x == 'F023':
    return 1
   return 0

inspect_vio_df['code_islarge'] = df_violate.violation_code.apply(is_large)
inspect_vio_df.head()
#df_violate.violation_code.apply(is_large)


# Feature: service code

# In[29]:


def is_routine(x):
    if x == 1:
     return 1
    return 0

inspect_vio_df['service_code'] = inspect_vio_df.service_code.apply(is_routine)


# In[30]:


def is_A(x):
    if x == 'A':
     return 1
    return 0

inspect_vio_df['grade'] = inspect_vio_df.grade.apply(is_A)       
    


# Extract Features and target from the large dataset

# In[33]:


columns = ['fac_type', 'risk','service_code','grade','month','day','year','code_islarge','score']
model_data = inspect_vio_df[columns]
model_data = model_data.dropna()
model_data.head()


# In[34]:


# Initialize H2O cluster
import h2o
h2o.init()


# In[36]:


#model data summary
model_data = H2OFrame(model_data)

for name in ['fac_type', 'risk','service_code','grade','month','day','year','code_islarge']:
    model_data[name] = model_data[name].asfactor()

model_data.summary()


# In[37]:


# Split into 70% training and 30% test dataset
strat_split = model_data['score'].stratified_split(test_frac=0.3, seed=42)
#train, test, valid = model_data.split_frame([0.6, 0.2]) #0.6,0.2,0.2
train = model_data[strat_split == 'train']
test = model_data[strat_split == 'test']

# Define features and target
feature = ['fac_type', 'risk','service_code','month','day','year','code_islarge']
target = 'grade'


# In[38]:


#gradient boosting machine(GBM)
from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution="multinomial",
          ntrees=100, max_depth=3, min_rows=2, learn_rate=0.2,seed=42)
model_gbm.train(x=feature, y=target, training_frame=train, validation_frame=test)


# In[80]:


model_gbm.score_history()


# In[73]:


# Feature importance
importance = model_gbm.varimp(use_pandas=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='scaled_importance', y='variable', data=importance)
plt.show()


# In[81]:


train_true = train.as_data_frame()['grade'].values
test_true = test.as_data_frame()['grade'].values
train_pred = model_gbm.predict(train).as_data_frame()['p1'].values
test_pred = model_gbm.predict(test).as_data_frame()['p1'].values

train_fpr, train_tpr, _ = roc_curve(train_true, train_pred)
test_fpr, test_tpr, _ = roc_curve(test_true, test_pred)
train_auc = np.round(auc(train_fpr, train_tpr), 3)
test_auc = np.round(auc(test_fpr, test_tpr), 3)


# In[82]:


# Classification report
print(classification_report(y_true=test_true, y_pred=(test_pred > 0.5).astype(int)))


# In[83]:


train_fpr = np.insert(train_fpr, 0, 0)
train_tpr = np.insert(train_tpr, 0, 0)
test_fpr = np.insert(test_fpr, 0, 0)
test_tpr = np.insert(test_tpr, 0, 0)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(train_auc))
ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(test_auc))
ax.plot(train_fpr, train_fpr, 'k--', label='Chance Curve')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.grid(True)
ax.legend(fontsize=12)
plt.show()


# In[110]:


cols = ['fac_type', 'risk','service_code','month','day','year','code_islarge']
_ = model_gbm.partial_plot(data=model_data, cols=cols, nbins=200, figsize=(18, 20))


# In[39]:


#GLM
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
glm_logistic = H2OGeneralizedLinearEstimator(family = "binomial")
glm_logistic.train(x=feature, y=target, training_frame=train, validation_frame=test, 
 model_id="glm_logistic")


# In[40]:


train_true = train.as_data_frame()['grade'].values
test_true = test.as_data_frame()['grade'].values
train_pred = glm_logistic.predict(train).as_data_frame()['p1'].values
test_pred = glm_logistic.predict(test).as_data_frame()['p1'].values

train_fpr, train_tpr, _ = roc_curve(train_true, train_pred)
test_fpr, test_tpr, _ = roc_curve(test_true, test_pred)
train_auc = np.round(auc(train_fpr, train_tpr), 3)
test_auc = np.round(auc(test_fpr, test_tpr), 3)


# In[41]:


# Classification report
print(classification_report(y_true=test_true, y_pred=(test_pred > 0.5).astype(int)))


# In[86]:


train_fpr = np.insert(train_fpr, 0, 0)
train_tpr = np.insert(train_tpr, 0, 0)
test_fpr = np.insert(test_fpr, 0, 0)
test_tpr = np.insert(test_tpr, 0, 0)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(train_auc))
ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(test_auc))
ax.plot(train_fpr, train_fpr, 'k--', label='Chance Curve')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.grid(True)
ax.legend(fontsize=12)
plt.show()


# In[109]:


cols = ['fac_type', 'risk','service_code','month','day','year','code_islarge']
_ = glm_logistic.partial_plot(data=model_data, cols=cols, nbins=200, figsize=(18, 20))


# In[93]:


#random forest
train, test = model_data.split_frame(ratios=[0.7])
rf_model =  H2ORandomForestEstimator(balance_classes=True, ntrees=100, mtries=-1, stopping_rounds=5, 
                                 stopping_metric='auc', score_each_iteration=True, seed=42)
rf_model.train(x=feature, y=target, training_frame=train, validation_frame=test)


# In[94]:


rf_model.score_history()


# In[95]:


#Make predictions
train_true = train.as_data_frame()['grade'].values
test_true = test.as_data_frame()['grade'].values
train_pred = rf_model.predict(train).as_data_frame()['p1'].values
test_pred = rf_model.predict(test).as_data_frame()['p1'].values

train_fpr, train_tpr, _ = roc_curve(train_true, train_pred)
test_fpr, test_tpr, _ = roc_curve(test_true, test_pred)
train_auc = np.round(auc(train_fpr, train_tpr), 3)
test_auc = np.round(auc(test_fpr, test_tpr), 3)


# In[96]:


print(classification_report(y_true=test_true, y_pred=(test_pred > 0.5).astype(int)))


# In[97]:


train_fpr = np.insert(train_fpr, 0, 0) 
train_tpr = np.insert(train_tpr, 0, 0) 
test_fpr = np.insert(test_fpr, 0, 0) 
test_tpr = np.insert(test_tpr, 0, 0) 
fig, ax = plt.subplots(figsize=(8, 6)) 
ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(train_auc)) 
ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(test_auc)) 
ax.plot(train_fpr, train_fpr, 'k--', label='Chance Curve') 
ax.set_xlabel('False Positive Rate', fontsize=12) 
ax.set_ylabel('True Positive Rate', fontsize=12) 
ax.grid(True) 
ax.legend(fontsize=12) 
plt.show()


# In[98]:


cols = ['fac_type', 'risk','service_code','month','day','year','code_islarge']
_ = rf_model.partial_plot(data=model_data, cols=cols, nbins=200, figsize=(18, 20))


# In[99]:


#NaiveBayes
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
train, test = model_data.split_frame(ratios=[0.7])
naive_bayes =  H2ONaiveBayesEstimator(nfolds=10, seed=42)
naive_bayes.train(x=feature, y=target, training_frame=train, validation_frame=test)


# In[100]:


#Make predictions
train_true = train.as_data_frame()['grade'].values
test_true = test.as_data_frame()['grade'].values
train_pred = naive_bayes.predict(train).as_data_frame()['p1'].values
test_pred = naive_bayes.predict(test).as_data_frame()['p1'].values

train_fpr, train_tpr, _ = roc_curve(train_true, train_pred)
test_fpr, test_tpr, _ = roc_curve(test_true, test_pred)
train_auc = np.round(auc(train_fpr, train_tpr), 3)
test_auc = np.round(auc(test_fpr, test_tpr), 3)


# In[101]:


print(classification_report(y_true=test_true, y_pred=(test_pred > 0.5).astype(int)))


# In[102]:


train_fpr = np.insert(train_fpr, 0, 0) 
train_tpr = np.insert(train_tpr, 0, 0) 
test_fpr = np.insert(test_fpr, 0, 0) 
test_tpr = np.insert(test_tpr, 0, 0) 
fig, ax = plt.subplots(figsize=(8, 6)) 
ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(train_auc)) 
ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(test_auc)) 
ax.plot(train_fpr, train_fpr, 'k--', label='Chance Curve') 
ax.set_xlabel('False Positive Rate', fontsize=12) 
ax.set_ylabel('True Positive Rate', fontsize=12) 
ax.grid(True) 
ax.legend(fontsize=12) 
plt.show()


# In[103]:


cols = ['fac_type', 'risk','service_code','month','day','year','code_islarge']
_ = naive_bayes.partial_plot(data=model_data, cols=cols, nbins=200, figsize=(18, 20))


# In[104]:


#deep learning
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
train, test = model_data.split_frame(ratios=[0.7])
neural_network =  H2ODeepLearningEstimator(seed=42)
neural_network.train(x=feature, y=target, training_frame=train, validation_frame=test)


# In[105]:


#Make predictions
train_true = train.as_data_frame()['grade'].values
test_true = test.as_data_frame()['grade'].values
train_pred = neural_network.predict(train).as_data_frame()['p1'].values
test_pred = neural_network.predict(test).as_data_frame()['p1'].values

train_fpr, train_tpr, _ = roc_curve(train_true, train_pred)
test_fpr, test_tpr, _ = roc_curve(test_true, test_pred)
train_auc = np.round(auc(train_fpr, train_tpr), 3)
test_auc = np.round(auc(test_fpr, test_tpr), 3)


# In[106]:


print(classification_report(y_true=test_true, y_pred=(test_pred > 0.5).astype(int)))


# In[107]:


train_fpr = np.insert(train_fpr, 0, 0) 
train_tpr = np.insert(train_tpr, 0, 0) 
test_fpr = np.insert(test_fpr, 0, 0) 
test_tpr = np.insert(test_tpr, 0, 0) 
fig, ax = plt.subplots(figsize=(8, 6)) 
ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(train_auc)) 
ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(test_auc)) 
ax.plot(train_fpr, train_fpr, 'k--', label='Chance Curve') 
ax.set_xlabel('False Positive Rate', fontsize=12) 
ax.set_ylabel('True Positive Rate', fontsize=12) 
ax.grid(True) 
ax.legend(fontsize=12) 
plt.show()


# In[108]:


cols = ['fac_type', 'risk','service_code','month','day','year','code_islarge']
_ = neural_network.partial_plot(data=model_data, cols=cols, nbins=200, figsize=(18, 20))


# In[111]:


# Shutdown h2o 
h2o.cluster().shutdown()

