#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


array=np.arange(10,20)
array


# In[3]:


plt.plot(array)


# In[4]:


fig=plt.figure()


# In[5]:


fig=plt.figure(figsize=(8,6))


# In[6]:


fig=plt.figure(figsize=(8,6))
ax1=fig.add_subplot(2,2,1)


# In[10]:


fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)


# In[12]:


fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)
ax1.plot([1,2,3,4,5])


# In[13]:


fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)
ax1.plot([1,2,3,4,5])
ax2.plot(np.random.randn(20).cumsum())


# In[14]:


fig, ax=plt.subplots(3,3)


# In[15]:


fig, ax=plt.subplots(3,3,figsize=(8,6))
ax[0,0].plot([1,2,3,4,5])


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
fig,ax =plt.subplots(3,3,figsize=(8,6))
ax[0,0].plot([1,2,3,4,5])
ax[2,2].plot([1,2,3,4,5])


# In[17]:


fig, ax=plt.subplots(3,3,figsize=(8,6))
ax[0,0].plot([1,2,3,4,5])
ax[2,2].plot([1,2,3,4,5])
fig.subplots_adjust(hspace=0.5,wspace=0.5)


# In[18]:


y=np.random.randn(20).cumsum()
y


# In[19]:


plt.plot(y)


# In[20]:


x=np.arange(0,200,10)
x


# In[21]:


plt.plot(x)


# In[22]:


plt.plot(x,y)


# In[23]:


plt.plot(x,y,color='r')


# In[24]:


plt.plot(x,y,color='green',marker='o')


# In[25]:


plt.plot(x,y,color='yellow',marker='s')


# In[29]:


plt.plot(x,y,color='green',marker='v',linestyle=':')


# In[34]:


plt.plot(x,y,'mv--')


# In[36]:


data=np.random.randn(200).cumsum()
data
fig, ax=plt.subplots(1,1)
ax.plot(data)


# In[37]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks(range(0,201,10))


# In[38]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks(range(0,201,50))


# In[39]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks([0,50,100,150,200])


# In[43]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks([0,50,100,150,200])
labels=ax.set_xticklabels(['one','two','three','four','five'])


# In[44]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks([0,50,100,150,200])
labels=ax.set_xticklabels(['one year sales','two year sales','three year sales','four year sales','five year sales'])


# In[45]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks([0,50,100,150,200])
labels=ax.set_xticklabels(['one year sales','two year sales','three year sales','four year sales','five year sales'],rotation=60)


# In[47]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks([0,50,100,150,200])
labels=ax.set_xticklabels(['one year sales','two year sales','three year sales','four year sales','five year sales'],rotation=60)
ax.set_xlabel('cumulative sum')


# In[48]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks([0,50,100,150,200])
labels=ax.set_xticklabels(['one year sales','two year sales','three year sales','four year sales','five year sales'],rotation=60)
ax.set_xlabel('cumulative sum')
ax.set_title('cumulative sum for random numbers')


# In[68]:


fig, ax=plt.subplots(1,1)
ax.plot(data)
ticks=ax.set_xticks([0,50,100,150,200])
labels=ax.set_xticklabels(['one year sales','two year sales','three year sales','four year sales','five year sales'],rotation=60)
ax.set_xlabel('cumulative sum')
ax.set_title('cumulative sum for random numbers')
ax.set_ylabel('values')


# # Adding legends

# In[58]:


data1=np.random.rand(200).cumsum()
data2=np.random.rand(200).cumsum()
data3=np.random.rand(200).cumsum()


# In[59]:


fig, ax=plt.subplots(1,1)
ax.plot(data1)
ax.plot(data2)
ax.plot(data3)


# In[60]:


fig, ax=plt.subplots(1,1)
ax.plot(data1,label='data1')
ax.plot(data2,label='data2')
ax.plot(data3,label='data3')


# In[61]:


fig, ax=plt.subplots(1,1)
ax.plot(data1,label='data1')
ax.plot(data2,label='data2')
ax.plot(data3,label='data3')
ax.legend()


# In[62]:


fig, ax=plt.subplots(1,1)
ax.plot(data1,label='data1')
ax.plot(data2,label='data2')
ax.plot(data3,label='data3')
ax.legend(loc='best')


# In[63]:


fig, ax=plt.subplots(1,1)
ax.plot(data1,label='data1')
ax.plot(data2,label='data2')
ax.plot(data3,label='data3')
ax.legend(loc='lower right')


# In[64]:


fig, ax=plt.subplots(1,1)
ax.plot(data1,label='data1')
ax.plot(data2,label='data2')
ax.plot(data3,label='data3')
ax.legend(loc='lower left')


# In[65]:


fig, ax=plt.subplots(1,1)
ax.plot(data1,label='data1')
ax.plot(data2,label='data2')
ax.plot(data3,label='data3')
ax.legend(loc='upper left')


# In[69]:


x=np.arange(0,10,0.1)
y=np.sin(x)


# In[71]:


fig, ax=plt.subplots(1,1)
ax.plot(x,y)


# In[72]:


fig, ax=plt.subplots(1,1)
ax.plot(x,y)
ax.text(4,0.7,'y=sin(x)',fontsize=15)


# In[77]:


fig, ax=plt.subplots(1,1)
ax.plot(x,y)
ax.text(0.5,0.5,'y=sin(x)',fontsize=15,transform=ax.transAxes)


# In[78]:


fig, ax=plt.subplots(1,1)
ax.plot(x,y)
ax.text(0.4,0.5,'y=sin(X)',fontsize=20,transform=ax.transAxes)


# In[85]:


fig, ax=plt.subplots(1,1)
ax.plot(x,y)
ax.arrow(0.5,1,0.6,0,width=0.05)
ax.arrow(6.8,1,0.6,0,width=0.05)


# In[108]:


fig, ax=plt.subplots(1,1)
ax.plot(x,y)
ax.text(0.35,0.8,'WaveLenght',fontsize=15)
ax.arrow(1.8,1,1.8,0,width=0.05)
ax.arrow(7.4,1,-2,0,width=0.05)


# In[110]:


x=np.arange(6)
y=np.array([23,34,65,78,51,55])
fig, ax=plt.subplots(1,1)
ax.plot(x,y,'ko--')


# In[112]:


fig, ax=plt.subplots(1,1)
ax.plot(x,y,'ko--')
ax.annotate('maximum',xy=(x[3],y[3]),xytext=(3,50),arrowprops=dict(facecolor='green',shrink=0.1))


# In[115]:


fig, ax=plt.subplots(1,1)
ax.plot(x,y,'ko--')
ax.annotate('maximum',xy=(x[3],y[3]),xytext=(3,50),arrowprops=dict(facecolor='green',shrink=0.1))
ax.annotate('minimum',xy=(x[0],y[0]),xytext=(0,50),arrowprops=dict(facecolor='green',shrink=0.1))


# In[118]:


fig, ax=plt.subplots(1,1)
circle=plt.Circle((0.5,0.5),0.2,fc='y')
ax.add_patch(circle)


# In[119]:


fig, ax=plt.subplots(1,1)
rectangle=plt.Rectangle((0.3,0.5),0.5,0.2,fc='y')
ax.add_patch(rectangle)


# In[120]:


fig, ax=plt.subplots(1,1)
tri=plt.Polygon([[0.6,0.4],[0.4,0.4],[0.4,0.6]],color='g',alpha=0.5)
ax.add_patch(tri)


# In[127]:


x=np.arange(6)
y=np.array([55,87,33,98,64,12])
fig, ax=plt.subplots(1,1)
ax.plot(x,y,'ko--')
ax.set_title('Data')
ax.set_xlabel('row')
ax.set_ylabel('column')
plt.savefig('ahmed/ehab.svg')


# In[132]:


x=np.arange(6)
y=np.array([55,87,33,98,64,12])
fig, ax=plt.subplots(1,1)
ax.plot(x,y,'ko--')
ax.set_title('Data')
ax.set_xlabel('row')
ax.set_ylabel('column')
plt.savefig('ahmed/mohamed.png')


# In[129]:


from os import listdir


# In[130]:


listdir('ahmed')


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


ser1=pd.Series(np.random.randn(10).cumsum())
ser1


# In[4]:


ser1.plot()


# In[9]:


data=pd.read_csv('all_wc_18_players_fifa.csv')
data.head()


# In[10]:


data.plot()


# In[11]:


data.plot(subplots=True)


# In[15]:


data.plot(subplots=True,layout=(1,5))


# In[17]:


data.plot(subplots=True,layout=(1,5),figsize=(12,6))


# In[18]:


data.plot(subplots=True,layout=(1,5),figsize=(12,6),legend=False)


# In[20]:


data.plot(subplots=True,layout=(1,5),figsize=(12,6),legend=False,title='informations of players')


# In[28]:


data.plot(subplots=True,layout=(1,5),figsize=(12,6),legend=False,title='Information of Players',sharey=True)


# In[25]:


data1=pd.read_csv('wizard.csv')
data1


# In[29]:


data1.plot(subplots=True,layout=(1,4),figsize=(12,6),legend=False,title='Information of Players',sharey=True)


# In[30]:


data.head()


# In[31]:


data.age.plot()


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


data=pd.Series([67,90,50,22,18],index=list('ABCDE'))
data


# In[4]:


data.plot.bar()


# In[9]:


data.plot.bar(rot=0,title='ahmed',xlabel='index',ylabel='data',color='red',alpha=0.7)


# In[14]:


data.plot.barh(color='green',alpha=0.7)


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


data=pd.read_csv('wizard.csv')
data.head()


# In[3]:


data.plot.bar(rot=0)


# In[4]:


data.plot.barh()


# In[5]:


data.plot.barh(legend=False)


# In[9]:


data.plot.bar(stacked=True,rot=0)


# In[10]:


data1=pd.read_csv('all_wc_18_players_fifa.csv')
data1.head()


# In[14]:


sns.barplot(data=data1)


# In[19]:


sns.barplot(data=data1[['caps','height','age']])


# In[22]:


sns.barplot(data=data1[['caps','height','age']],ci=False)


# In[25]:


sns.barplot(x=data1.position,y=data1.weight,ci=False)


# In[31]:


sns.barplot(x=data1.weight,y=data1.position,orient='h',ci=False)


# In[33]:


sns.barplot(x=data1.number,y=data1.weight,hue=data1.position)


# In[39]:


data=pd.read_csv('wizard.csv')
data


# In[52]:


data['storm'].plot.hist(bins=5)


# In[54]:


data['storm'].plot.hist(bins=5,color='red',edgecolor='yellow',linewidth=1)


# In[57]:


data1['age'].plot.hist(bins=12,color='brown',edgecolor='red',linewidth=2)


# In[58]:


data1['height'].plot.hist(bins=12,color='green',edgecolor='blue',linewidth=2)


# In[60]:


data1['weight'].plot.density()


# In[61]:


sns.displot(data1['age'],bins=12,color='black')


# In[62]:


sns.displot(data1['caps'],kind='kde',color='violet')


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
data1=pd.read_csv('all_wc_18_players_fifa.csv')
data1.head()


# In[3]:


sns.regplot(x='height',y='weight',data=data1)


# In[4]:


sns.regplot(x='height',y='weight',data=data1,ci=False)


# In[5]:


sns.regplot(x='height',y='weight',data=data1,ci=False,fit_reg=False)


# In[6]:


sns.pairplot(data1)


# In[7]:


sns.pairplot(data1,diag_kind='kde')


# In[8]:


sns.catplot(x='position',y='number',data=data1)


# In[9]:


sns.catplot(x='position',y='number',data=data1,kind='bar')


# In[10]:


sns.catplot(x='position',y='number',data=data1,kind='bar',ci=False)


# In[13]:


sns.catplot(x='position',y='number',data=data1,kind='box',ci=True)


# In[22]:


data2=pd.read_csv('wizard.csv')
data2


# In[20]:


sns.catplot(x='death',y='Subscription',hue='Marital Status',data=data2,kind='bar')


# In[1]:


from datetime import datetime
import pandas as pd


# In[2]:


time_now=datetime.now()
time_now


# In[3]:


time_now.year


# In[4]:


time_now.day


# In[6]:


time_now.second


# In[12]:


time1=datetime(2001,2,18)
time2=datetime(2002,2,28)


# In[13]:


time3=time2-time1
time3


# In[18]:


time3.days


# In[19]:


from datetime import timedelta


# In[20]:


time1+timedelta(12)


# In[21]:


time2-timedelta(16)


# In[1]:


import pandas as pd
from datetime import datetime


# In[2]:


data='2021,2,18'
data


# In[3]:


data1=datetime.strptime(data,'%Y,%m,%d')


# In[4]:


data1


# In[8]:


time_str='2006-05-15'
time_str


# In[11]:


time_str1=datetime.strptime(time_str,'%Y-%m-%d')
time_str1


# In[12]:


data=pd.read_csv('metadata.csv')
data.head()


# In[15]:


data['date'].dtype


# In[18]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[19]:


data2=pd.DataFrame([['30Apr19'],['29May19'],['28June19'],['27July19'],['26Apr20'],['20June20'],['18Feb21'],['29Mar21']],columns=['date'])
data2


# In[20]:


data2['date'].dtype


# In[21]:


data2['date']=pd.to_datetime(data2['date'])
data2['date']


# In[22]:


data2


# In[23]:


data2=data2.set_index('date')


# In[20]:


data=pd.read_csv('metadata.csv',parse_dates=['date'],index_col='date')
data.head()


# In[21]:


import numpy as np
import pandas as pd


# In[22]:


data2.loc['2021']


# In[26]:


data2.loc['2019':'2020']


# In[27]:


data2.index.is_unique


# In[35]:


wizard=pd.read_csv('wizard.csv')
wizard


# In[40]:


wizard1=pd.date_range(start='2001/2/18',end='2001/2/25')
wizard1


# In[41]:


w=wizard.set_index(wizard1)
w


# In[44]:


wizard2=pd.date_range(start='2001/5/13',periods=15)
wizard2


# In[2]:


import pandas as pd
wizard3=pd.date_range(start='2001/01/01',end='2001/12/30',freq='BM')
wizard3


# In[5]:


wizard4=pd.date_range(start='2001/01/01',end='2001/12/30',freq='B')
wizard4


# In[6]:


wizard5=pd.date_range(start='2001/01/01',end='2001/12/30',freq='4h')
wizard5


# In[12]:


wizard5=pd.date_range(start='2001/01/01',end='2001/12/30',freq='WOM-3FRI')
wizard5


# In[14]:


import numpy as np
array_1=np.array([[1,2,7],[3,4,8]])
array_2=np.array([[1,2],[3,9],[4,16]])
np.dot(array_1,array_2)


# In[15]:


for i in [0,2,3]:
    if i%2==0:
        print("even")
    elif i==0:
        print("zero")
    else:
        print("odd")
      


# In[16]:


x=0
y=x%2
print(y)


# In[24]:


data2


# In[29]:


data2.shift(periods=1)


# In[38]:


data2=pd.DataFrame([['30Apr19',133.5],['29May19',55.8],['28June19',75.3],['27July19',43.5],['26Apr20',97.3],['20June20',76.8],['18Feb21',25.6],['29Mar21',98.5]]
                   ,columns=['date','apples'])
data2


# In[39]:


data2.set_index('date',inplace=True)
data2                                                                                                                                                                                                                                                                                                                                                                                                       


# In[40]:


data2.shift(periods=1)


# In[41]:


data2.shift(periods=2)


# In[42]:


data2.shift(periods=3)


# In[43]:


data2.shift(-1)


# In[45]:


data2['oranges']=data2['apples'].shift(1)


# In[46]:


data2


# In[47]:


data2.dropna(inplace=True)


# In[48]:


data2


# In[49]:


import pytz


# In[54]:


pytz.common_timezones


# In[56]:


index=pd.date_range(start='2019/2/18 13:30',periods=12,freq='H')
index


# In[57]:


print(index.tz)


# In[58]:


index2=pd.date_range(start='2021/6/10',periods=10,freq='H',tz='UTC')


# In[59]:


index2


# In[60]:


index2=pd.date_range(start='2021/6/10',periods=10,freq='H',tz='US/Mountain')


# In[61]:


index2


# In[62]:


data2


# In[7]:


data3=pd.read_csv('metadata.csv')
data3


# In[4]:


data3.set_index('date',inplace=True)
data3                        


# In[80]:


print(data3.index.tz)


# In[81]:


local=data3.tz_localize('UTC')


# In[82]:


local


# In[84]:


local.tz_convert('Europe/Brussels')


# In[1]:


import numpy as np
import pandas as pd


# In[18]:


data3=pd.DataFrame([['30Apr19',133.5],['29May19',55.8],['28June19',75.3],['27July19',43.5],['26Apr20',97.3],['20June20',76.8],['18Feb21',25.6],['29Mar21',98.5]]
                   ,columns=['date','apples'])
data3


# In[19]:


data3['date']=pd.to_datetime(data3['date'])
data3


# In[20]:


data3.set_index('date',inplace=True)
data3                        


# In[21]:


data3.resample('w').mean()


# In[22]:


data3.resample('m').mean()


# In[23]:


data3.resample('m',kind='period').mean()


# In[24]:


data3.resample('m',kind='period').sum()


# In[38]:


data3


# In[39]:


data3['apples'].plot()


# In[40]:


data3.rolling(window=3).mean()


# In[52]:


data3['apples'].plot()
data3['apples'].rolling(window=3).mean().plot()


# In[54]:


data3['apples'].plot(figsize=(8,5))


# In[56]:


data3['apples'].plot(figsize=(3,3))


# In[ ]:




