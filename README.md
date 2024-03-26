# AI_Project_2

### Note for Cody and David...
Check out 'Encoder Tool.py' and 'ml_pipelines 3 test scaler in line.ipynb'.  In Encoder, imports a junk dataset (Main section after functions at bottom of file). Application presents column types and value counts so you can choose the encoding type.  For OHE, it allows you to specify the order of the values (example is education, etc.).  Check out the 'Possible To Do's below.

Update from 9:00 PM Sunday, 3/24.  Check out the Streamlit app. I am deploying as a web application. I'll show you two tomorrow. MUCH nicer interface. The web framework I selected is pretty easy to use!! Talk soon.


### Assigned: Use classification methods and supervised leaning to model a substancial dataset (will include 'Churn' analysis). Adding: Generate tools for rapid data modeling


### Possible To Do's
* Get Encoding working - DF
* Encoder needs a max_categories add for OHE and possibly label encoding. - DF
* there is no grid search build in to optimize models - B priority for now
* Cody - Need to integrate the ml_pipeline code into the py file -> Cody [ml_pipelines 3 test scaler in line. In streamlit app, put in #4]
* Imputation automation is not build
* DF: check out web frameworks weekend of 3/22. I'm not fond of the 'teminal applications' (Muy Fea, muy fea...). Likely out of scope and not enough time, but I'll take a look out of curiousity.
* DF: [Note: this is all B priority] Exploratory Data Analysis: Integrate 3D graphing (B priority) and integrate into python library (for pip installs). Extend 3D package for 2D graphs.
* Cody For EDA -> Y versus any other other other columns 
* Markdown report of models, scoring, parameters, B Priority
* Preprocessing now includes several encoders and StandardScaling. We should add Target Encoding into the 'Encoder Tool'. Also needs first pass preprocess or. I'll add as separate item.
* Pre-encoding Preprocessor: What columns are numeric, but are currently object? Is a missing character code in the initial dataframe that needs to be replaced with pd.null, etc.?
* Why not have the program do a full 'markup' report including.
    * processing steps used
    * which columns were encoded which way
    * parameter summary
    * models used
    * model scoring: Need to review what I have so far on this. It's basic, but does some nice extended model scoring. Code is not in the repo. I'll send out a separate repo, since it is in the format of a python libary for a pip install.
    * grid search param and results
    * Exporatory Data Analysis (EDA):  Select 2D, 3D graphs, covariance matrix
* Add freely :)  Would be nice to have some reusable work completed for rapid data modeling beyond our class!
