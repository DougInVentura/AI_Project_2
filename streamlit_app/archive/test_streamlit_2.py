
import pandas as pd
import numpy as np
import os


df_test_1 = pd.read_csv("streamlit_app/data/churn.csv")

# print("Here is the original dataframe...")
# print(f"num columns: {df_test_1.shape}")
# print(df_test_1.iloc[0:10,0:9])
# print(df_test_1.iloc[:10:,9:17])
 
# the_output = df_test_1[["payment_method"]].groupby("payment_method").size().reset_index(name='counts').sort_values(by='counts', ascending=False)

max_cat_dict = {"payment_meth_2":["other_pay",2],
                "service_calls_text2":["more than 5",6]}



# print("number_customer_service_calls val counts")
# print(df_test_1["number_customer_service_calls"].value_counts())
df_test_1["service_calls_text"] = df_test_1["number_customer_service_calls"].astype(str)
df_test_1["service_calls_text2"] = df_test_1["service_calls_text"]
df_test_1["payment_meth_2"] = df_test_1["payment_method"]
df_test_1.to_csv("streamlit_app/data/churn_max_cat_test")
df_after = df_test_1.copy()
# print(df_test_1[["number_customer_service_calls","service_calls_text"]].head())
# print("\n------------- ")
# print("the column          name_of_other            number of other")
for the_col in max_cat_dict:
    name_of_other = max_cat_dict[the_col][0]
    max_categories = max_cat_dict[the_col][1]
    # print(f"{the_col:25} -> {name_of_other:20}     {max_categories}    type of max_cat: {type(max_categories)}")
    df_cat = df_test_1[[the_col]].groupby(the_col).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    df_cat = df_cat.reset_index(drop=True)
    # print("df cat head is below")
    # print(df_cat.head(10))
    mapping = {}
    # print("----- Detail for the column")
    print("for column: ",the_col)
    for i, row in df_cat.iterrows():
        # print(f"i={i}  level: {row[0]} and row[count]: {row[1]}")
        if i > max_categories - 1:
            mapping[row[0]] = name_of_other
        else:
            mapping[row[0]] = row[0]
    print(f"mapping dir for {the_col}: \n {mapping}")
    df_after[the_col] = df_after[the_col].map(mapping)

print(" service call text               payment_method")
print(df_after[["service_calls_text","service_calls_text2","payment_method","payment_meth_2"]].head(40))

    # df_after = df_after[the_col].map(mapping)

# print(df_after)