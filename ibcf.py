#########################################################################

## 커스텀 > pandas dataframe 으로 변환

df = as_pandas(cursor)

# In[5]:

df.head()

# In[9]:

df['page_meta_id'].nunique(), df['pay_account_id'].nunique()

# In[10]:

# full matrix

view_matrix = df.pivot(values='view_cnt',index='page_meta_id',columns='pay_account_id')
view_matrix

# In[11]:

from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = view_matrix.copy().fillna(0)

# In[13]:

matrix_dummy

# In[17]:

# 출력물 제한없이

pd.set_option('display.max_rows',20) # 없애려면 none
pd.set_option('display.max_columns',20)

# In[18]:

item_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
item_similarity = pd.DataFrame(item_similarity, index=view_matrix.index, columns=view_matrix.index)

# In[19]:

item_similarity

# In[21]:

item_similarity = pd.DataFrame(item_similarity, index = view_matrix.index, columns = view_matrix.index)
item_similarity

# In[28]:

user_matrix = df.pivot(values='view_cnt',index='pay_account_id',columns='page_meta_id')
user_matrix = user_matrix.copy().fillna(0)

# In[29]:

user_matrix

# In[26]:

def predict(ratings, similarity):
pred = ratings.dot(similarity) / np.array([np.abs(similarity.sum(axis=1))])
return pred

# In[37]:

item_prediction = predict(user_matrix, item_similarity)
item_prediction

# In[31]:

#item_prediction = item_prediction.reset_index()
#item_prediction

# In[38]:

item_prediction.info()

# In[33]:

item_prediction[item_prediction['pay_account_id'] == '952559']

# In[61]:

# 이미 평가한 것은 제외

def remove_rated_items(ratings, predictions):
rated_items = (ratings != 0)
predictions[rated_items] = -np.inf
return predictions

updated_predictions = remove_rated_items(user_matrix, item_prediction)

# In[62]:

updated_predictions

# In[63]:

#updated_predictions = updated_predictions.reset_index()
updated_predictions

# In[59]:

updated_predictions[updated_predictions['pay_account_id'] == '952559']

# In[64]:

max_scores = updated_predictions.idxmax(axis=1)
max_scores

# In[69]:

type(max_scores)

# In[71]:

max_scores.index, max_scores.values

# In[65]:

pay_account_id = pd.Series(max_scores.index, name='pay_account_id')
pay_account_id

# In[66]:

page_meta_id = pd.Series(max_scores.values, name='page_meta_id')
page_meta_id

# In[67]:

df_result = pd.DataFrame({
'pay_account_id': pay_account_id,
'page_meta_id': page_meta_id
})

df_result

# In[55]:

df_result.info()

# In[68]:

df_result[df_result['pay_account_id']=='952559']

# In[78]:

# top3

top_3_items = updated_predictions.apply(lambda row: pd.Series(row.nlargest(3).index), axis=1)
top_3_items.columns = ['1st', '2nd', '3rd']

# In[83]:

top_3_items = top_3_items.reset_index()
top_3_items

# In[84]:

top_3_items[top_3_items['pay_account_id'] == '952559']

# In[75]:

top_3_items.index

# In[76]:

top_3_items.values
