
# __Programming for Analytics: Final Project__

### Arunima Grover, Rubal Shrestha, Francheska Orellana

#### This part (until _ Section 1_) needs to be run everytime


```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
#import seaborn #as sns
import copy 
%matplotlib inline
rcParams['figure.figsize'] = 15, 6
```


```python
#Read in the file and check 
df = pd.read_csv("Mass_Shootings_Dataset_Ver_5.csv", encoding = 'latin-1',index_col=0)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Location</th>
      <th>Date</th>
      <th>Incident Area</th>
      <th>Open/Close Location</th>
      <th>Target</th>
      <th>Cause</th>
      <th>Summary</th>
      <th>Fatalities</th>
      <th>Injured</th>
      <th>Total victims</th>
      <th>Policeman Killed</th>
      <th>Age</th>
      <th>Employeed (Y/N)</th>
      <th>Employed at</th>
      <th>Mental Health Issues</th>
      <th>Race</th>
      <th>Gender</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
    <tr>
      <th>S#</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Texas church mass shooting</td>
      <td>Sutherland Springs, TX</td>
      <td>11/5/2017</td>
      <td>Church</td>
      <td>Close</td>
      <td>random</td>
      <td>unknown</td>
      <td>Devin Patrick Kelley, 26, an ex-air force offi...</td>
      <td>26</td>
      <td>20</td>
      <td>46</td>
      <td>0.0</td>
      <td>26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>White</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Walmart shooting in suburban Denver</td>
      <td>Thornton, CO</td>
      <td>11/1/2017</td>
      <td>Wal-Mart</td>
      <td>Open</td>
      <td>random</td>
      <td>unknown</td>
      <td>Scott Allen Ostrem, 47, walked into a Walmart ...</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
      <td>White</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Edgewood businees park shooting</td>
      <td>Edgewood, MD</td>
      <td>10/18/2017</td>
      <td>Remodeling Store</td>
      <td>Close</td>
      <td>coworkers</td>
      <td>unknown</td>
      <td>Radee Labeeb Prince, 37, fatally shot three pe...</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0.0</td>
      <td>37</td>
      <td>NaN</td>
      <td>Advance Granite Store</td>
      <td>No</td>
      <td>Black</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Las Vegas Strip mass shooting</td>
      <td>Las Vegas, NV</td>
      <td>10/1/2017</td>
      <td>Las Vegas Strip Concert outside Mandala Bay</td>
      <td>Open</td>
      <td>random</td>
      <td>unknown</td>
      <td>Stephen Craig Paddock, opened fire from the 32...</td>
      <td>59</td>
      <td>527</td>
      <td>585</td>
      <td>1.0</td>
      <td>64</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Unclear</td>
      <td>White</td>
      <td>M</td>
      <td>36.181271</td>
      <td>-115.134132</td>
    </tr>
    <tr>
      <th>5</th>
      <td>San Francisco UPS shooting</td>
      <td>San Francisco, CA</td>
      <td>6/14/2017</td>
      <td>UPS facility</td>
      <td>Close</td>
      <td>coworkers</td>
      <td>NaN</td>
      <td>Jimmy Lam, 38, fatally shot three coworkers an...</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>0.0</td>
      <td>38</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>M</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Title', 'Location', 'Date', 'Incident Area', 'Open/Close Location',
           'Target', 'Cause', 'Summary', 'Fatalities', 'Injured', 'Total victims',
           'Policeman Killed', 'Age', 'Employeed (Y/N)', 'Employed at',
           'Mental Health Issues', 'Race', 'Gender', 'Latitude', 'Longitude'],
          dtype='object')




```python
#Takes care of any issues possible when calling the columns
df.columns = ['Title', 'Location', 'DateT', 'IncidentArea', 'LocationOC',
       'Target', 'Cause', 'Summary', 'Fatalities', 'Injured', 'TotalVictims',
       'PolicemanKilled', 'Age', 'Employeed(Y/N)', 'EmployedAt',
       'MentalHealthIssues', 'Race', 'Gender', 'Latitude', 'Longitude']
df.columns
```




    Index(['Title', 'Location', 'DateT', 'IncidentArea', 'LocationOC', 'Target',
           'Cause', 'Summary', 'Fatalities', 'Injured', 'TotalVictims',
           'PolicemanKilled', 'Age', 'Employeed(Y/N)', 'EmployedAt',
           'MentalHealthIssues', 'Race', 'Gender', 'Latitude', 'Longitude'],
          dtype='object')




```python
#To check the different types of Locations
df.LocationOC.unique()
```




    array(['Close', 'Open', nan, 'Open+Close', 'Open+CLose'], dtype=object)




```python
#Cleaning data, combining Open+Close and Open+CLose 
df.loc[df.LocationOC=='Open+CLose','LocationOC']='Open+Close'
df.dropna(subset = ['LocationOC'], inplace = True)
#df.dropna(thresh = 2)
df.LocationOC.unique()
```




    array(['Close', 'Open', 'Open+Close'], dtype=object)



#### Everything up to here needs to be run everytime. The sections below are independent so one does not need to be run for the other to run.

## Section 1: Locations(Open/Close) and Target

### Data Preparation: Location(Open/Close) and Target


```python
#Create new dataframe (deep copy) 
#Call it dfTarget because for this section, the focus will be on comparing with Target types
dfTarget=copy.deepcopy(df)
#First, remove rows with empty cells
dfTarget.dropna(subset = ['Target'], inplace = True)
dfTarget.Target.unique()
```




    array(['random', 'coworkers', 'women', 'police', 'Family',
           'uninvited guests', 'birthday party bus', 'Trooper', 'party guests',
           'neighbors', 'club members', 'Policeman', 'Family/Neighbors',
           'drug dealer', 'protestors', 'Students', 'Ex-Wife', 'Coworkers',
           'Ex-Girlfriend', 'Marines', 'Ex-girlfriend', 'House Owner',
           'Friends', 'Contestant', 'Ex-Girlfriend & Family',
           'Ex-Wife & Family', 'Ex-Girlfriend+random', 'rapper+random',
           'TSA Officer', "partner's family", 'Girlfriend',
           "Coworker's Family", 'Family+students', 'Ex-Coworkers', 'Sikhs',
           'Congresswoman', 'Policeman+Council Member', 'Students+Teachers',
           'school girls', 'Ex-GirlFriend', 'hunters', 'Teachers',
           'Students+Parents', 'psychologist+psychiatrist', 'lawyers',
           'Social Workers', 'monks', 'Children', 'postmaster',
           'welding shop employees'], dtype=object)



#### _In this part, we organize Target into categories_


```python
#Category 1: Officials
#Officials include police, Trooper, Policeman, Marines, TSA Officer, Congresswoman and Policeman+Council Member 

Off_list = ['police', 'Trooper', 'Policeman', 'Marines', 'TSA Officer', 'Congresswoman', 'Policeman+Council Member']
for i in range (0, len(Off_list)):
    dfTarget.loc[dfTarget.Target==Off_list[i],'Target']='Officials'

dfTarget.Target.unique()
```




    array(['random', 'coworkers', 'women', 'Officials', 'Family',
           'uninvited guests', 'birthday party bus', 'party guests',
           'neighbors', 'club members', 'Family/Neighbors', 'drug dealer',
           'protestors', 'Students', 'Ex-Wife', 'Coworkers', 'Ex-Girlfriend',
           'Ex-girlfriend', 'House Owner', 'Friends', 'Contestant',
           'Ex-Girlfriend & Family', 'Ex-Wife & Family',
           'Ex-Girlfriend+random', 'rapper+random', "partner's family",
           'Girlfriend', "Coworker's Family", 'Family+students',
           'Ex-Coworkers', 'Sikhs', 'Students+Teachers', 'school girls',
           'Ex-GirlFriend', 'hunters', 'Teachers', 'Students+Parents',
           'psychologist+psychiatrist', 'lawyers', 'Social Workers', 'monks',
           'Children', 'postmaster', 'welding shop employees'], dtype=object)




```python
#Category 2: Ethnicity, Race, Religion, Gender (ERRG)
#ERRG includes black men, Sikhs, monks, prayer group, women 

ERRG_list = ['black men', 'Sikhs', 'monks', 'prayer group', 'women']
for i in range (0, len(ERRG_list)):
    dfTarget.loc[dfTarget.Target==ERRG_list[i],'Target']='Ethnicity_Race_Regilion_Gender'

dfTarget.Target.unique()
```




    array(['random', 'coworkers', 'Ethnicity_Race_Regilion_Gender',
           'Officials', 'Family', 'uninvited guests', 'birthday party bus',
           'party guests', 'neighbors', 'club members', 'Family/Neighbors',
           'drug dealer', 'protestors', 'Students', 'Ex-Wife', 'Coworkers',
           'Ex-Girlfriend', 'Ex-girlfriend', 'House Owner', 'Friends',
           'Contestant', 'Ex-Girlfriend & Family', 'Ex-Wife & Family',
           'Ex-Girlfriend+random', 'rapper+random', "partner's family",
           'Girlfriend', "Coworker's Family", 'Family+students',
           'Ex-Coworkers', 'Students+Teachers', 'school girls',
           'Ex-GirlFriend', 'hunters', 'Teachers', 'Students+Parents',
           'psychologist+psychiatrist', 'lawyers', 'Social Workers',
           'Children', 'postmaster', 'welding shop employees'], dtype=object)




```python
#Category 3: School 
#School includes Students, school girls, Teachers, Students+Parents, Children, and Students+Teachers

s_list = ['Students', 'school girls', 'Teachers', 'Students+Parents', 'Children', 'Students+Teachers']
for i in range (0, len(s_list)):
    dfTarget.loc[dfTarget.Target==s_list[i],'Target']='School'
    
dfTarget.Target.unique()
```




    array(['random', 'coworkers', 'Ethnicity_Race_Regilion_Gender',
           'Officials', 'Family', 'uninvited guests', 'birthday party bus',
           'party guests', 'neighbors', 'club members', 'Family/Neighbors',
           'drug dealer', 'protestors', 'School', 'Ex-Wife', 'Coworkers',
           'Ex-Girlfriend', 'Ex-girlfriend', 'House Owner', 'Friends',
           'Contestant', 'Ex-Girlfriend & Family', 'Ex-Wife & Family',
           'Ex-Girlfriend+random', 'rapper+random', "partner's family",
           'Girlfriend', "Coworker's Family", 'Family+students',
           'Ex-Coworkers', 'Ex-GirlFriend', 'hunters',
           'psychologist+psychiatrist', 'lawyers', 'Social Workers',
           'postmaster', 'welding shop employees'], dtype=object)




```python
#Category 4: Family & Friends (FF) 
#FF includes 

ff_list = ['Family', 'Family/Neighbors', 'Ex-Wife', 'Ex-Girlfriend','Ex-GirlFriend','Ex-girlfriend', 'Ex-Girlfriend & Family', 'Ex-Wife & Family', 'Ex-Girlfriend+random','Family+random', "partner's family", 'Girlfriend', 'Family+students', 'Friends', 'neighbors'] 
for i in range (0, len(ff_list)):
    dfTarget.loc[dfTarget.Target==ff_list[i],'Target']='Family_Friends'
    
dfTarget.Target.unique()
```




    array(['random', 'coworkers', 'Ethnicity_Race_Regilion_Gender',
           'Officials', 'Family_Friends', 'uninvited guests',
           'birthday party bus', 'party guests', 'club members', 'drug dealer',
           'protestors', 'School', 'Coworkers', 'House Owner', 'Contestant',
           'rapper+random', "Coworker's Family", 'Ex-Coworkers', 'hunters',
           'psychologist+psychiatrist', 'lawyers', 'Social Workers',
           'postmaster', 'welding shop employees'], dtype=object)




```python
#Category 5: Work 
#Work includes coworkers, Coworkers, Coworker's Family, Ex-Coworkers 

work_list = ['coworkers', 'Coworkers', "Coworker's Family", 'Ex-Coworkers'] 
for i in range (0, len(work_list)):
    dfTarget.loc[dfTarget.Target==work_list[i],'Target']='Work'
    
dfTarget.Target.unique()
```




    array(['random', 'Work', 'Ethnicity_Race_Regilion_Gender', 'Officials',
           'Family_Friends', 'uninvited guests', 'birthday party bus',
           'party guests', 'club members', 'drug dealer', 'protestors',
           'School', 'House Owner', 'Contestant', 'rapper+random', 'hunters',
           'psychologist+psychiatrist', 'lawyers', 'Social Workers',
           'postmaster', 'welding shop employees'], dtype=object)




```python
#Category 6: Party  
#Party includes uninvited guests, birthday party bus, party guests

party_list = ['uninvited guests', 'birthday party bus', 'party guests'] 
for i in range (0, len(party_list)):
    dfTarget.loc[dfTarget.Target==party_list[i],'Target']='Party'
    
dfTarget.Target.unique()
```




    array(['random', 'Work', 'Ethnicity_Race_Regilion_Gender', 'Officials',
           'Family_Friends', 'Party', 'club members', 'drug dealer',
           'protestors', 'School', 'House Owner', 'Contestant',
           'rapper+random', 'hunters', 'psychologist+psychiatrist', 'lawyers',
           'Social Workers', 'postmaster', 'welding shop employees'], dtype=object)




```python
#Category 7: Other Specific Groups (OSG))  
#OSG includes protestors, hunters, basketball players, psychologist+psychiatrist, lawyers, Social Workers, club members, welding shop employees

OSG_list = ['protestors', 'hunters', 'basketball players', 'psychologist+psychiatrist', 'lawyers', 'Social Workers', 'club members', 'welding shop employees'] 
for i in range (0, len(OSG_list)):
    dfTarget.loc[dfTarget.Target==OSG_list[i],'Target']='Other_Specific_Groups'
    
dfTarget.Target.unique()
```




    array(['random', 'Work', 'Ethnicity_Race_Regilion_Gender', 'Officials',
           'Family_Friends', 'Party', 'Other_Specific_Groups', 'drug dealer',
           'School', 'House Owner', 'Contestant', 'rapper+random', 'postmaster'], dtype=object)




```python
#Category 8: Other
#Other includes drug dealer, House Owner, Contestant, postmaster 

Other_list = ['drug dealer', 'House Owner', 'Contestant', 'postmaster'] 
for i in range (0, len(Other_list)):
    dfTarget.loc[dfTarget.Target==Other_list[i],'Target']='Other'
    
dfTarget.Target.unique()
```




    array(['random', 'Work', 'Ethnicity_Race_Regilion_Gender', 'Officials',
           'Family_Friends', 'Party', 'Other_Specific_Groups', 'Other',
           'School', 'rapper+random'], dtype=object)




```python
#Category 9: Random
#Random includes random and rapper+random 

Rand_list = ['random', 'rapper+random'] 
for i in range (0, len(Rand_list)):
    dfTarget.loc[dfTarget.Target==Rand_list[i],'Target']='Random'
    
dfTarget.Target.unique()
```




    array(['Random', 'Work', 'Ethnicity_Race_Regilion_Gender', 'Officials',
           'Family_Friends', 'Party', 'Other_Specific_Groups', 'Other',
           'School'], dtype=object)



### Data Analysis: Location(Open/Close) and Target

#### _ Table, Bar Graph, & Pie Chart_


```python
#Group the 2 columns Location OC and Target by common values and put it in TGroup1; then unstack
TGroup1=dfTarget.groupby(['LocationOC','Target'])
LOC_by_TGroup1 = TGroup1.size().unstack()
LOC_by_TGroup1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Target</th>
      <th>Ethnicity_Race_Regilion_Gender</th>
      <th>Family_Friends</th>
      <th>Officials</th>
      <th>Other</th>
      <th>Other_Specific_Groups</th>
      <th>Party</th>
      <th>Random</th>
      <th>School</th>
      <th>Work</th>
    </tr>
    <tr>
      <th>LocationOC</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Close</th>
      <td>3.0</td>
      <td>43.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>71.0</td>
      <td>33.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>Open</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>51.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Open+Close</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create a reg plot and using matplotlib & edit it
lt1=LOC_by_TGroup1.plot(kind='bar', figsize=[16,8], title='Targets by Location Type (Open/Close)', colormap='Spectral')
lt1.set_ylabel('Number of Shootings')
lt1.set_xlabel('Location Type')
plt.xticks(rotation=0)
lt1.legend(loc='best')
plt.show()
```


![png](output_25_0.png)



```python
#Same thing but this time group in opposite formation (to graph later)
TGroup2=dfTarget.groupby(['Target','LocationOC'])
LOC_by_TGroup2 = TGroup2.size().unstack()
LOC_by_TGroup2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>LocationOC</th>
      <th>Close</th>
      <th>Open</th>
      <th>Open+Close</th>
    </tr>
    <tr>
      <th>Target</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ethnicity_Race_Regilion_Gender</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Family_Friends</th>
      <td>43.0</td>
      <td>7.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Officials</th>
      <td>3.0</td>
      <td>9.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Other_Specific_Groups</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Party</th>
      <td>7.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Random</th>
      <td>71.0</td>
      <td>51.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>School</th>
      <td>33.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Work</th>
      <td>26.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Graph 
lt2=LOC_by_TGroup2.plot(kind='bar', figsize=[16,8],title='Location Type (Open/Close) by Target', colormap='summer')
lt2.set_xlabel('Target')
lt2.set_ylabel('Number of Shootings')
plt.xticks(rotation=45)
lt2.legend(loc='best')
plt.show()
```


![png](output_27_0.png)



```python
#Stacking bars to view it differently
lt2Stacked=LOC_by_TGroup2.plot(kind='bar', figsize=[16,8], stacked=True,title='Location Type (Open/Close) by Target', colormap='summer')
lt2Stacked.set_xlabel('Target')
lt2Stacked.set_ylabel('Number of Shootings')
plt.xticks(rotation=45)
lt2Stacked.legend(loc='best')
plt.show()
```


![png](output_28_0.png)



```python
#Pie chart
labels=('ERRG','FF','Officials','Other','OSG','Party','Random','School','Work')
explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.2)
Plt4=LOC_by_TGroup2.plot(kind='pie',explode=explode,labels=labels,labeldistance=1.05, figsize=[21.25,12.25], subplots=True,autopct='%2.1f%%',title='Target by Location Type (Open/Close)',counterclock=False, colormap='nipy_spectral')
plt.xticks(rotation=60)
plt.subplots_adjust(left=0.11, bottom=0.2, right=0.9)
plt.legend(loc='upper right')
plt.show()
```


![png](output_29_0.png)


## Section 2: Location(Open/Close) and Cause

### Data Preparation: Locations(Open/Close) and Cause


```python
dfCause=copy.deepcopy(df)
#First, remove rows with empty cells
dfCause.dropna(subset = ['Cause'], inplace = True)
dfCause.Cause.unique()
```




    array(['unknown', 'terrorism', 'unemployement', 'racism', 'frustration',
           'domestic dispute', 'anger', 'psycho', 'revenge',
           'domestic disputer', 'suspension', 'religious radicalism',
           'failing exams', 'robbery'], dtype=object)




```python
#To avoid any syntax errors 
dfCause.loc[dfCause.Cause=='domestic dispute','Cause']='domestic_dispute'
dfCause.loc[dfCause.Cause=='domestic disputer','Cause']='domestic_dispute'
dfCause.loc[dfCause.Cause=='religious radicalism','Cause']='religious_radicalism'
dfCause.loc[dfCause.Cause=='failing exams','Cause']='failing_exams'
dfCause.Cause.unique()
```




    array(['unknown', 'terrorism', 'unemployement', 'racism', 'frustration',
           'domestic_dispute', 'anger', 'psycho', 'revenge', 'suspension',
           'religious_radicalism', 'failing_exams', 'robbery'], dtype=object)



### Data Analysis: Location(Open/Close) and Cause

#### _ Table, Bar Graph & Pie Chart _ 


```python
#Grouping columns again
CGroup1=dfCause.groupby(['LocationOC', 'Cause'])
LOC_by_CGroup1=CGroup1.size().unstack()
LOC_by_CGroup1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Cause</th>
      <th>anger</th>
      <th>domestic_dispute</th>
      <th>failing_exams</th>
      <th>frustration</th>
      <th>psycho</th>
      <th>racism</th>
      <th>religious_radicalism</th>
      <th>revenge</th>
      <th>robbery</th>
      <th>suspension</th>
      <th>terrorism</th>
      <th>unemployement</th>
      <th>unknown</th>
    </tr>
    <tr>
      <th>LocationOC</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Close</th>
      <td>25.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>39.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>44.0</td>
      <td>8.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Open</th>
      <td>16.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Open+Close</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Same concept as before with different data (Cause)
lc1=LOC_by_CGroup1.plot(kind='bar', figsize=[16,8],title='Cause by Location Type (Open/Close)', colormap='Spectral')
lc1.set_ylabel('Number of Shootings')
lc1.set_xlabel('Location Type')
plt.xticks(rotation=0)
lc1.legend(loc='best')
plt.show()
```


![png](output_37_0.png)



```python
#switching x and y
CGroup2=dfCause.groupby([ 'Cause', 'LocationOC'])
LOC_by_CGroup2=CGroup2.size().unstack()
LOC_by_CGroup2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>LocationOC</th>
      <th>Close</th>
      <th>Open</th>
      <th>Open+Close</th>
    </tr>
    <tr>
      <th>Cause</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>anger</th>
      <td>25.0</td>
      <td>16.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>domestic_dispute</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>failing_exams</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>frustration</th>
      <td>12.0</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>psycho</th>
      <td>39.0</td>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>racism</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>religious_radicalism</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>revenge</th>
      <td>8.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>robbery</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>suspension</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>terrorism</th>
      <td>44.0</td>
      <td>14.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>unemployement</th>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#graphing switched x and y
lc2=LOC_by_CGroup2.plot(kind='bar', figsize=[16,8],title='Location Type (Open/Close) by Cause', colormap='summer')
lc2.set_xlabel('Cause')
lc2.set_ylabel('Number of Shootings')
plt.xticks(rotation=45)
lc2.legend(loc='best')
plt.show()

```


![png](output_39_0.png)



```python
#Stacked Graph
lc2Stacked=LOC_by_CGroup2.plot(kind='bar', stacked=True, figsize=[16,8],title='Location Type (Open/Close) by Cause', colormap='summer')
lc2Stacked.set_xlabel('Cause')
lc2Stacked.set_ylabel('Number of Shootings')
plt.xticks(rotation=45)
lc2Stacked.legend(loc='best')
plt.show()

```


![png](output_40_0.png)



```python
# Pie chart
labels=('Anger','Domestic dispute','Failing exams','Frustation','Psycho','Racism','Religious.R','revenge','robbery','suspension','terrorism','unemployment','unknown')
explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.1)
plt5=LOC_by_CGroup2.plot(kind='pie',explode=explode, figsize=[21.25,12.25],labels=labels,labeldistance=1.05, subplots=True,autopct='%1.1f%%',title='Cause by Location Type (Open/Close)',counterclock=False, colormap='nipy_spectral')
plt.legend(loc='upper right')
plt.subplots_adjust(left=0.11, bottom=0.2, right=0.9)
plt.xticks(rotation=60)
plt.show()
```


![png](output_41_0.png)


## Section 3: Location(Open/Close) Time Series

### Data Preparation: Location(Open/Close) Time Series


```python
#Preparing data for Time Series
import datetime
import dateutil
from datetime import datetime
from dateutil import parser

dfDateAll=copy.deepcopy(df)

drop_list = [ 'Title', 'Location', 'IncidentArea', 'Target','Cause', 'Summary' ,'Fatalities', 'Injured', 'TotalVictims','PolicemanKilled','Age', 'Employeed(Y/N)', 'EmployedAt', 'MentalHealthIssues', 'Race', 'Gender', 'Latitude', 'Longitude']
for i in range (0, len(drop_list), 1):
    del dfDateAll[drop_list[i]]

```


```python
#Changing the dates to keep only the year since data will be analyzed by year 
dfDateAll.DateT = pd.to_datetime(dfDateAll.DateT).dt.strftime('%Y')
dfDateAll.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateT</th>
      <th>LocationOC</th>
    </tr>
    <tr>
      <th>S#</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>Close</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>Open</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>Close</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>Open</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>Close</td>
    </tr>
  </tbody>
</table>
</div>




```python
#To confirm data is being grouped by year
dfDateAll.groupby('DateT')['LocationOC'].count()
```




    DateT
    1966     2
    1971     1
    1974     2
    1976     2
    1979     2
    1982     2
    1983     2
    1984     3
    1985     2
    1986     3
    1987     1
    1988     6
    1989     3
    1991     5
    1992     3
    1993     9
    1994     4
    1995     4
    1996     2
    1997     5
    1998     5
    1999     5
    2000     1
    2001     2
    2002     2
    2003     3
    2004     2
    2005     3
    2006     5
    2007    10
    2008     6
    2009     8
    2010     2
    2011     6
    2012    14
    2013    15
    2014    13
    2015    56
    2016    64
    2017    10
    Name: LocationOC, dtype: int64



### Data Analysis: Location(Open/Close) Time Series


```python
#Create a table to organize LocationOC by Year 
byYear=dfDateAll.groupby(['DateT','LocationOC'])
count_byYear=byYear.size().unstack()
TS=count_byYear
TS.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>LocationOC</th>
      <th>Close</th>
      <th>Open</th>
      <th>Open+Close</th>
    </tr>
    <tr>
      <th>DateT</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1966</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Time Series
TS.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17dff3a8860>




![png](output_49_1.png)


#### End of Notebook


```python

```
