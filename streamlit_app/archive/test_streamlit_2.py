import streamlit as st
import pandas as pd
import numpy as np

df_test_1 = pd.DataFrame({'animal':['cat', 'dog', 'rat', 'cat', 'dog'],
                   'animal_name':['ari','coco','willard','ava','bear'],
                   'rand_num':[10,20,30,40,50],
                   'weight': [15,30,1,16, 70],
                   'breed':['alley','mut','NYC rat','alley','golden'],
                   'education':['high school','college','primary','high school','post grad']})

st.write("Here is the original dataframe...")
st.dataframe(df_test_1)
st.write("--------------------------------------")

column_list = df_test_1.columns


df_column_actions = pd.DataFrame({'the_col':df_test_1.columns,'the_types':df_test_1.dtypes}).sort_values(by = 'the_types').reset_index(drop=True)
df_column_actions['encoding_num'] = 1
df_column_actions['max_categories'] = 1

st.write(f"shape of df_column_actions is: {np.shape(df_column_actions)}")

# Initialize session state for comments if it doesn't exist
if 'comments' not in st.session_state:
    st.session_state.comments = [''] * len(df_column_actions)

# Function to update the DataFrame with comments
def update_comments():
    df_column_actions['Comments'] = st.session_state.comments

# Layout with columns: DataFrame on the left, comments on the right
col1, col2 = st.columns(2)

with col1:
    st.write("DataFrame:")
    st.dataframe(df_column_actions)

with col2:
    st.write("Comments:")
    for i in range(len(df_column_actions)):
        st.session_state.comments[i] = st.text_input(f"Row {i+1}", key=f"comment_{i}",  value=st.session_state.comments[i])

# Update button and logic
if st.button('Update Comments'):
    update_comments()
    st.success('Comments updated in the DataFrame')
    # Refresh the display
    col1, col2 = st.columns(2)
    with col1:
        st.write("DataFrame:")
        st.dataframe(df_column_actions)