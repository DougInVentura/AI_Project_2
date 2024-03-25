# test 1
import streamlit as st
import pandas as pd
import numpy as np


df = pd.DataFrame({
    'x': np.arange(10),  # Example x values
    'y': np.random.randn(10)  # Example y values
})

# define a test dataframe

df_test_1 = pd.DataFrame({'animal':['cat', 'dog', 'rat', 'cat', 'dog'],
                   'animal_name':['ari','coco','willard','ava','bear'],
                   'rand_num':[10,20,30,40,50],
                   'weight': [15,30,1,16, 70],
                   'breed':['alley','mut','NYC rat','alley','golden'],
                   'education':['high school','college','primary','high school','post grad']})




# Initialize session state for comments if it doesn't exist
if 'comments' not in st.session_state:
    st.session_state.comments = [''] * len(df)

# Function to update the DataFrame with comments
def update_comments():
    df['Comments'] = st.session_state.comments

# Display the DataFrame and comments input fields
st.write('DataFrame:')
for i in range(len(df)):
    row = df.iloc[i]
    st.write(f"Row {i+1}: {row['x']}, {row['y']}")
    st.session_state.comments[i] = st.text_input(f"Comment for row {i+1}", key=f"comment_{i}", value=st.session_state.comments[i])

# Update button
if st.button('Update Comments'):
    update_comments()
    st.success('Comments updated in the DataFrame')

# Optionally display the updated DataFrame
st.write(df)

st.write("--------------------------------------")
st.write

# Set the title of the web app
st.title('#### My first Streamlit app')

# Set a subheader
st.subheader('Line Chart')

# Create a line chart
st.line_chart(df.rename(columns={'x': 'index'}).set_index('index'))