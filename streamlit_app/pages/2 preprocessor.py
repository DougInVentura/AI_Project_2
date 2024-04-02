import streamlit as st
import os
import pandas as pd
import numpy as np
import json
import datetime
import pdb


st.set_page_config(
    page_title="Preprocessor",
    page_icon="👋",
)

st.write("""## 👋 **Dataframe Preprocessor Page** 👋
### Supervised Machine Learning Classification preprocessing, encoding and model building and scoring)""")

# --------  Functions  ----------------------------------------------------
def do_consolidate_categories(X_df):
    st.write('write this function')
    st.session_state['Consolidating_Categories'] = True
    # define an editable dataframe w column number and max categies for Object columns
    return False

    return True

# -------- Main Main  Main ------------------------------------------------

if ('df_loaded' in st.session_state) and (st.session_state.df_loaded):
    # should have total dataframe and X_in_process_df and y_df ready (when df_loaded is True)
    st.write("df loaded is set.")
    df_in_process = st.session_state['df_initial']
    X_in_process_df = st.session_state['X_in_process_df']
    y_df = st.session_state['y_df']

    st.write('df in process loaded as well as X_in_process_df and y_df')

    st.dataframe(X_in_process_df.head(8))
  

    column_list = X_in_process_df.columns
    obj_types = [X_in_process_df[col].dtype.name for col in X_in_process_df.columns]
    num_columns = len(column_list)
    false_array = np.repeat(False, num_columns)
    other_name_array = [pd.NA] * num_columns
    max_categories_array = [np.nan] * num_columns
   
    instruction_df = pd.DataFrame({"Column":column_list,
                                    "Type":obj_types,
                                    "Drop?":false_array, 
                                    "Max_Categories":max_categories_array,
                                    'Name_for_Other':other_name_array,
                                    "Perform_Mapping":false_array
                                    })
    
    
    st.markdown("Fill out preprocessing **Instruction Table**. Select action for any columns that is needed.")
    st.markdown("**Drop?** - Select if any of the columns are to be dropped.")
    st.markdown("**Max_Categories** - Any categories for the column beyond this number (in order of the value counts table) will be consolidated into 'other'")
    st.markdown("**Name_for_Other** - If Max_Categories is given a number, the category will be named the string in 'Name for Other'")
    st.markdown("**Perform Mapping** If selected, then mapping this will be performed after any dropping and consolidation")
    st.markdown("")       
    st.write("""Order of actions: 1) Drop columns, 2) Category Consolidation (implement max categories). 3) Name consolidated categories (Name_for_Other)
and finally, perform mapping""")
    with st.popover("Click for value counts of the columns"):
        st.markdown("Value Counts for the Columns... 👋")
        for the_col in X_in_process_df.columns:
            st.write(f"Column: {the_col}")
            st.write(X_in_process_df[the_col].value_counts())
            
    if ('have_edited_instr_df' in st.session_state) and st.session_state['have_edited_instr_df']:
        df_2_use = st.session_state['edited_instr_df']
    else:
        df_2_use = instruction_df.copy()

    edited_instr_df = st.data_editor(df_2_use,
        column_config={
            "Drop?": st.column_config.CheckboxColumn(
                "Drop this column?",
                help="select checkbox to drop column",
                default=False
            ),
            "Perform_Mapping": st.column_config.CheckboxColumn(
                "Perform Mapping?",
                help="After any column dropping and consolidating infrequent categories (Max_Elements), perform mapping?",
                default=False
            )
        },
        disabled=["Column","Type"],
        hide_index=True,
    )
    

    if st.button("Fill out instruction table, then :blue[Click] when complete"):
        drop_col = []
        consolidate_catgegories = []
        perform_mapping_list = []
        for index, the_row in edited_instr_df.iterrows():
            if the_row['Drop?']:
                drop_col.append(the_row['Column'])
            else:
                # dont perform these steps if the column is being dropped
                max_cat = the_row["Max_Categories"]
                if max_cat is not pd.NA and max_cat > 1:
                    other_name_canidate = the_row['Name_for_Other']
                    if len(other_name_canidate) > 0:
                        name_for_other = other_name_canidate
                    else:
                        name_for_other = 'Other'
                    consolidate_catgegories.append({the_row['Column']:[name_for_other,max_cat]})
                perform_mapping = the_row['Perform_Mapping']
                if perform_mapping:
                    perform_mapping_list.append(the_row['Column'])
        # Past looping over the rows. we have the lists for dropping, consolidating catefgoreies, and mapping
        # start with delete
        if len(drop_col) > 0:
            X_in_process_df = X_in_process_df.drop(columns = drop_col)
            st.write('#### **After column drop(s)**')
        else:
            st.write('#### **No columns to drop in X dataframe**')
        col1, col2 = st.columns([1,3])
        with col1:
            st.write("#### **y dataframe**")
            st.dataframe(y_df)
        with col2:
            st.write("#### **X dataframe**")
            st.dataframe(X_in_process_df)
            
        st.session_state['X_in_process_df'] = X_in_process_df
        if 'dir_and_X_IN_PROCESS_fname' in st.session_state:  
            dir_and_X_IN_PROCESS_fname = st.session_state['dir_and_X_IN_PROCESS_fname']
            X_in_process_df.to_csv(dir_and_X_IN_PROCESS_fname, index = False)
        else:
            st.write("dir and file name was not passed from 'Select data file...'. Using data\X_temp.csv instead")
            X_in_process_df.to_csv("data/temp.csv", index = False)
        st.write(f"Preprocessing has been completed.")
        st.session_state['Preprocessing_complete'] = True
else:
    st.write("Go back to 'Select Data File...'.  Nothing is loaded")

if 'Preprocessing_complete' in st.session_state and st.session_state['Preprocessing_complete']:
    if st.button("Preprocessing is complete. :blue[Click here] to goto 'Select Encoding Strategy"):
        st.switch_page("pages/3 Select Encoding Strategy.py")


  
         

