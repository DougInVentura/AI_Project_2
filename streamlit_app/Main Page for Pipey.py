import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("""# 👋 Welcome to Pipey 👋
### Supervised Machine Learning Classification preprocessing, encoding and model building and scoring
### Project 2 ASU AI Course: Cody, Doug, David
         
##### Version 0.2a """)

st.sidebar.success("Select a page to continue")

if st.button(":blue[Click Here] to select a csv file with your data"):
    st.switch_page("pages/1 Select data file and init proc.py")


