#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
ser1=pd.Series([23,54,np.nan,None])
ser1


# In[3]:


ser1.isnull()


# In[9]:


ser2=pd.Series(['green','black','white',None,'red'])
ser2


# In[10]:


ser2.isnull()


# In[11]:


ser1=pd.Series([23,54,np.nan,None,34,87])
ser1


# In[12]:


ser1.dropna()


# In[13]:


ser1


# In[16]:


ser1[ser1.notnull()]


# In[17]:


ser1.dropna(inplace=True)


# In[18]:


ser1


# In[20]:


dt1=pd.DataFrame([[1,None,3,4,5],[6,None,8,9,10],[11,12,13,14,15]])
dt1


# In[22]:


dt1.dropna()


# In[23]:


dt1.dropna(axis=1)


# In[24]:


dt2=pd.DataFrame([[6,None,8,9,10],[None,None,None,None,None],[11,12,13,14,15],[16,17,18,19,20]])
dt2


# In[25]:


dt2.dropna(how='all')


# In[26]:


dt3=pd.DataFrame([[6,None,np.nan,9,10],[1,None,2,3,4],[11,None,13,14,15],[16,np.nan,18,19,20]])
dt3


# In[27]:


dt3.dropna(axis=1,how='all')


# In[34]:


dt4=pd.DataFrame([[6,None,np.nan,9,10],[1,2,None,np.nan,np.nan],[11,None,13,14,15],[np.nan,16,18,19,20]],columns=(['white','blue','black','red','orange']))
dt4


# In[35]:


dt4.dropna(thresh=3)


# In[36]:


dt4


# In[37]:


dt4.fillna(20)


# In[42]:


dt4.fillna({'white':9,'blue':0,'black':7,'red':5,'orange':1})


# In[43]:


dt4.fillna(method='ffill')


# In[44]:


dt4.fillna(method='bfill')


# In[45]:


dt4.fillna(dt4.mean())


# In[69]:


dt1=pd.DataFrame([['Andy Allanson',293,66,1,30],['Alan Ashley',315,81,7,24],['Alvin Davis',479,130,18,66],['Andy Allanson',293,66,1,30],['Andre Dawson',496,141,20,65]],columns=(['Name','AtBot','hits','HmRun','Runs']))


# In[70]:


dt1


# In[71]:


dt1.duplicated()


# In[72]:


dt1.duplicated().any()


# In[73]:


dt1.drop_duplicates()


# In[74]:


dt1


# In[75]:


dt1.drop_duplicates(inplace=True)


# In[76]:


dt1


# In[77]:


temp=pd.Series([23,37,999,32,32,28,999,19,24])
temp


# In[79]:


temp.replace(999,np.nan,inplace=True)
temp


# In[82]:


temp=pd.Series([23,37,999,32,32,28,1000,19,20,-999,24])
temp


# In[84]:


temp.replace([999,-999,1000],np.nan,inplace=True)
temp


# In[85]:


gender=pd.DataFrame(['male','femal','male','male','female','mal','female'],index=list('abcdefg'),columns=['gender'])


# In[86]:


gender


# In[87]:


gender.replace({'mal':'male','femal':'female'})


# In[89]:


data=pd.DataFrame(np.arange(12).reshape((4,3)),index=['green','red','black','white'],columns=['one','two','three'])
data


# In[91]:


data.rename(index={'green':'yellow'},inplace=True)
data


# In[93]:


data.rename(columns={'three':'four'},inplace=True)
data


# In[94]:


data.index=data.index.str.upper()
data


# In[95]:


data.columns=data.columns.str.title()
data


# In[4]:


import numpy as np
import pandas as pd
data=pd.read_csv('all_wc_18_players_fifa.csv')
data.head(15)


# In[3]:


data.describe()


# In[7]:


data[data.height>199]


# In[10]:


data.loc[149,['height']]=199
data.loc[163,['height']]=199
data[data.height>199]
data[data.height>199].any()


# In[11]:


data=pd.read_csv('all_wc_18_players_fifa.csv')


# In[13]:


data.loc[149,['height']]=np.nan
data.loc[163,['height']]=np.nan


# In[16]:


data[data.height>199]
data.head(164)


# In[18]:


ser1=pd.Series(np.random.randint(20,size=10))
ser1


# In[24]:


ser1.sample(frac=1)


# In[25]:


ser1=ser1.sample(frac=1).reset_index(drop=True)
ser1


# In[26]:


data=pd.read_csv('all_wc_18_players_fifa.csv')


# In[27]:


data


# In[28]:


sample2=data.sample(frac=0.2)
sample2


# In[29]:


sample3=data.sample(frac=0.2).reset_index(drop=True)


# In[30]:


sample3


# In[32]:


sample4=data.sample(n=100).reset_index(drop=True)
sample4


# In[33]:


data=pd.read_csv('all_wc_18_players_fifa.csv')


# In[39]:


position_dummies=pd.get_dummies(data['position'])
position_dummies


# In[38]:


data_dummies=pd.get_dummies(data['team'])
data_dummies


# In[41]:


data_with_dummies=data.join(position_dummies)
data_with_dummies


# In[42]:


text1='jone,sam,jake'
text1.split(',')


# In[43]:


words=[x.strip() for x in text1.split(',')]
words


# In[53]:


text2='sam will go to the school today'
text2.split( )


# In[54]:


text3=['sam','yahoo.com']
'@'.join(text3)


# In[55]:


'school' in text2


# In[56]:


text2.index('school')


# In[57]:


text2.find('school')


# In[58]:


text2.find('ahmed')


# In[64]:


text2.count('to')


# In[65]:


text3='sam:jake:jone'
text3.replace(':',',')


# In[1]:


import numpy as np
import pandas as pd


# In[4]:


ser1=pd.Series(np.random.randn(9),index=[['2010','2010','2010','2011','2011','2011','2012','2012','2012']
                                         ,['one','two','three','one','two','three','one','two','three']])


# In[5]:


ser1


# In[6]:


ser1.index


# In[7]:


ser1['2010']


# In[11]:


ser1['2011':'2012']


# In[13]:


ser1.loc[['2010','2012']]


# In[14]:


ser1.loc[:,'one']


# In[16]:


data1=ser1.unstack()
data1


# In[18]:


data1.stack()


# In[19]:


data=pd.read_csv('all_wc_18_players_fifa.csv')
data.head()


# In[20]:


multi_data=data.set_index(['name','team'])
multi_data


# In[21]:


multi_data.index


# In[24]:


multi_data.loc[('Gabriel Mercado','Argentina')]


# In[25]:


multi_data.loc['Edinson Cavani']


# In[27]:


data=pd.DataFrame(np.random.randint(100,size=(4,4)),columns=[['green','green','black','black'],['one','two','one','two']])
data


# In[30]:


data=pd.DataFrame(np.random.randint(100,size=(6,4)),index=[['green','green','black','black','yellow','yellow']
                                                           ,['one','two','one','two','one','two']])
data


# In[33]:


data2=data.swaplevel(0,1)
data2


# In[35]:


data2.sort_index(level=0)


# In[38]:


data.swaplevel(0,1).sort_index(level=0)


# In[39]:


data=pd.DataFrame(np.random.randint(100,size=(6,4)),index=[['green','green','black','black','yellow','yellow']
                                                           ,['one','two','one','two','one','two']])
data


# In[40]:


data.mean()


# In[41]:


data.index.names


# In[44]:


data.index.names=['colors','numbers']
data


# In[46]:


data.mean(level='colors')


# In[47]:


data.mean(level='numbers')


# In[48]:


data.sum(level='colors')


# In[49]:


data.sum(level='numbers')


# In[50]:


data.columns


# In[65]:


football=pd.read_csv('all_wc_18_players_fifa.csv')
football.head()


# In[66]:


football.columns


# In[67]:


football=football.set_index('name')


# In[68]:


football.head()


# In[70]:


football3=football.set_index(['team','club'])
football3


# In[72]:


football.set_index('team',drop=False)


# In[73]:


football


# In[74]:


football3


# In[76]:


football3.reset_index()


# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data1=pd.DataFrame({'states':['california','Georgia','florida','arizona'],'population':[40,10,21,7]})
data1


# In[3]:


data2=pd.DataFrame({'states':['arizona','colorado','indiana','florida'],'area':[113,104,36,35]})
data2


# In[4]:


pd.merge(data1,data2)


# In[5]:


pd.merge(data1,data2,on='states')


# In[6]:


pd.merge(data1,data2,how='outer')


# In[7]:


pd.merge(data1,data2,how='left')


# In[8]:


pd.merge(data1,data2,how='right')


# In[11]:


data3=pd.DataFrame({'states_name':['california','georgia','florida','arizona'],'population':[40,10,21,7]})
data3


# In[12]:


data2


# In[13]:


pd.merge(data3,data2,left_on='states_name',right_on='states')


# In[24]:


data4=pd.DataFrame({'states':['california','georgia','florida','arizona'],'population':[40,10,21,7],'water':[4,6,7,2]})
data4


# In[25]:


data5=pd.DataFrame({'states':['arizona','colorado','indiana','florida'],'area':[113,104,36,65],'water':[2,8,3,7]})
data5


# In[26]:


pd.merge(data4,data5,on = 'states',how='outer')


# In[27]:


pd.merge(data4,data5,on = 'states',how='outer',suffixes=('_data4','_data5'))


# In[28]:


frame1=pd.DataFrame(np.random.randint(100,size=(4,3)),index=['b','c','e','f'],columns=['green','red','white'])
frame1


# In[29]:


frame2=pd.DataFrame(np.random.randint(100,size=(3,4)),index=['a','b','e'],columns=['blue','yellow','purple','black'])
frame2


# In[30]:


pd.merge(frame1,frame2,left_index=True,right_index=True)


# In[31]:


pd.merge(frame1,frame2,left_index=True,right_index=True,how='outer')


# In[33]:


ser1=pd.Series([1,2,3],index=['a','b','c'])
ser1


# In[34]:


ser2=pd.Series([4,5,6],index=['d','e','f'])
ser2


# In[36]:


pd.concat([ser1,ser2])


# In[37]:


ser3=pd.Series([7,8,9],index=['a','b','g'])
ser3


# In[38]:


pd.concat([ser1,ser3])


# In[39]:


pd.concat([ser1,ser3],axis=1)


# In[40]:


pd.concat([ser1,ser3],axis=1,join='inner')


# In[41]:


pd.concat([ser1,ser3],axis=1,join='inner')


# In[45]:


frame1=pd.DataFrame({'states':['california','georgia','florida','arizona'],'population':[40,10,21,7]})
frame1


# In[46]:


frame2=pd.DataFrame({'states':['hawaii','colorado','indiana','alaska'],'population':[1.5,10.4,5.7,0.7]})
frame2


# In[47]:


pd.concat([frame1,frame2])


# In[49]:


pd.concat([frame1,frame2],ignore_index=True)


# In[50]:


frame3=pd.DataFrame({'states':['california','georgia','florida','arizona'],'population':[40,10,21,7]})
frame3


# In[51]:


frame4=pd.DataFrame({'water':[23,54,12,45],'area':[113,104,36,65]})
frame4


# In[52]:


pd.concat([frame3,frame4],axis=1)


# In[62]:


data=pd.read_csv('all_wc_18_players_fifa.csv',index_col=['name','team'])
data


# In[63]:


data2=data.stack()
data2


# In[64]:


data2.unstack()


# In[65]:


data2.unstack(0)


# In[66]:


data2.unstack('name')


# In[70]:


frame1.unstack('name')


# In[71]:


frame2.stack()


# In[76]:


data1=pd.read_csv('all_wc_18_players_fifa.csv')
data1


# In[78]:


data2=data1.melt(id_vars=['birth_date','shirt_name','club','number','name','team','league','position'])
data2


# In[81]:


data2=data1.melt(id_vars=['birth_date','shirt_name','club','number','name','team','league','position'],
                var_name='recommended',value_name='values')
data2


# In[86]:


data3=pd.read_csv('Life expectancy.csv',index_col='Year')
data3


# In[87]:


data3.melt()


# In[89]:


data3.melt(ignore_index=False)


# In[90]:


data3.sort_index()


# In[91]:


data1=pd.read_csv('Life expectancy.csv')
data1


# In[93]:


data1.pivot(index='Year',columns='Entity',values='Life expectancy')


# In[94]:


data1.pivot('Year','Entity','Life expectancy')


# In[96]:


frame1=data1.set_index(['Year','Entity'])
frame1


# In[97]:


frame1.unstack('Entity')


# In[ ]:




