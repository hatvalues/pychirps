# Just for dev - map out how the app will work with hard-coded data
# In due time - it will be adaptable based on uploaded data and model
from data_preprocs.data_providers import cervicalb_pd as cvb

column_names = cvb.features.columns
spiel = cvb.spiel