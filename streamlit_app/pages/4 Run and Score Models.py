import streamlit as st
import os 
import io


if st.session_state['is_df_scaled']:
    df_scaled = st.session_state['df_scaled']
    if len(df_scaled) > 0:
        st.write(f"""Ready to go!  Scaled dataframe (df_scaled) beginning is below
                    {df_scaled.head(8)}
                 """)
