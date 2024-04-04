from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import pandas as pd
import streamlit as st

## Note: (Most) All of this code was dropped in from kanaries.net and we did not write any of it. Web Address: https://docs.kanaries.net/pygwalker/use-pygwalker-with-streamlit
 
# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Data Analysis using Pygwalker",
    layout="wide"
)
 
# Establish communication between pygwalker and streamlit
init_streamlit_comm()
 
# Add a title
st.title("Pygwalker Analysis Screen")
 
# Get an instance of pygwalker's renderer. You should cache this instance to effectively prevent the growth of in-process memory.
@st.cache_resource
def get_pyg_renderer(df) -> "StreamlitRenderer":
    # When you need to publish your app to the public, you should set the debug parameter to False to prevent other users from writing to your chart configuration file.
    return StreamlitRenderer(df, spec="./gw_config.json", debug=False)
 
if 'df_initial_loaded' in st.session_state and st.session_state['df_initial_loaded'] and 'df_initial' in st.session_state:
    df_initial = st.session_state['df_initial']
    renderer = get_pyg_renderer(df_initial)
    # Render your data exploration interface. Developers can use it to build charts by drag and drop.
    renderer.render_explore()
else:
    st.write("Go back to 'Select data file and init proc' and select your datafile to use Pygwalker")
 