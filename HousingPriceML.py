#!usr/bin/python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import tarfile
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import FunctionTransformer   #to add extra futures after creating 1hotencoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# to make this notebook's output stable across runs
np.random.seed(42)

# Where to save the figures
PROJECT_ROOT_DIR = "/home/savoroso/Desktop/Python_Project/my_scripts_repo/my_repo"
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
if not os.path.isdir(IMAGES_PATH):
	os.makedirs(IMAGES_PATH)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print (path)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)




DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def plot_hist(data_val_pd):
	data_val_pd.hist(bins=50,figsize=(20,15))
	save_fig('Attrib Hist')
	plt.show()
	


def train_test_func(data_val_pd):
	t_set,t_set = train_test_split(data_val_pd,test_size= 0.2,random_state= 42)
	return t_set,t_set


def train_test_func_stratified(data_val_pd):
	split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	for train_index, test_index in split.split(housing, housing["income_cat"]):
	    strat_train_set = housing.loc[train_index]
	    strat_test_set = housing.loc[test_index]


def add_extra_features(X,add_bedroom_per_room = True):
	rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
	population_per_household = X[:,population_ix]/X[:,household_ix]

	if add_bedroom_per_room:
		bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
		return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]

	else:
		return np.c_[X,rooms_per_household,population_per_household]


from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
	rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
	population_per_household = X[:, population_ix]/X[:, household_ix]
	if add_bedrooms_per_room:
		bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
		return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
	else:
		return np.c_[X, rooms_per_household, population_per_household]

#execute all funtions
def main():
### fetching data

	# fetch_housing_data()
	housing = load_housing_data()

### Exploring data to gain insights
	# print (housing.head())
	# print(housing.info())
	# print(housing['ocean_proximity'].value_counts())
	# print (housing.describe())
	# plot_hist(housing)

### Create train set and test set from data using random sampling; use sklearn to get create train and test set
	train_set,test_set = train_test_func(housing)

### Exploring test_set

	# print(test_set.head())
	# housing['median_income'].hist()
	# plt.show()

### To limit the income category, we will divide by 1.5
	housing['income_cat'] = np.ceil(housing['median_income']/1.5)  

### Generalize the label with minimal value, so those greater than 5 label it with 5.
	housing['income_cat'].where(housing['income_cat']<5,5.0,inplace = True)

	# housing['income_cat'].hist()
	# print(housing['income_cat'].value_counts())
	# plt.show()


### Create train and test set from data using stratified sampling
	split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	for train_index, test_index in split.split(housing, housing['income_cat']):
	    strat_train_set = housing.loc[train_index]
	    strat_test_set = housing.loc[test_index]

	# print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))
	# print(housing["income_cat"].value_counts() / len(housing))


	for set_ in (strat_train_set, strat_test_set):
		set_.drop("income_cat", axis=1, inplace=True)

	
### Discover and visualize the data to gain insights
# 	housing = strat_train_set.copy()
# 	housing.plot(kind="scatter",x ="longitude", y = "latitude",alpha = 0.1)
	# save_fig("better_visual_plot")

	# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 #    s=housing["population"]/100, label="population", figsize=(10,7),
 #    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
 #    sharex=False)
	# plt.legend()
	# save_fig("housing_prices_scatterplot")

	corr_matrix = housing.corr()
	# print(corr_matrix["median_house_value"].sort_values(ascending = False))


	attribs = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
	
	scatter_matrix(housing[attribs],figsize = (12,8))
	# save_fig("scatter_matrix_plot")


	housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
	plt.axis([0, 16, 0, 550000])
	# save_fig("income_vs_house_value_scatterplot")


	housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
	housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
	housing["population_per_household"]=housing["population"]/housing["households"]

	corr_matrix = housing.corr()
	# print(corr_matrix["median_house_value"].sort_values(ascending = False))



### Prepare data for machine learning algo

	housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
	housing_labels = strat_train_set["median_house_value"].copy()

	sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
	# print(sample_incomplete_rows)
	# print(sample_incomplete_rows.dropna(subset= ["total_bedrooms"]))  #drop all data with na
	# print(sample_incomplete_rows.drop("total_bedrooms",axis=1)) # drop column with na
	


	### impute the missing values
	try:
	    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
	except ImportError:
	    from sklearn.preprocessing import Imputer as SimpleImputer

	imputer = SimpleImputer(strategy="median")

	### removing categorical data
	housing_num = housing.drop('ocean_proximity', axis=1)
	imputer.fit(housing_num)
	# print (imputer.statistics_)

	###transform the training set:
	X = imputer.transform(housing_num)
	housing_tr = pd.DataFrame(X,columns = housing_num.columns,index = housing.index)

	housing_cat = housing[['ocean_proximity']]

	from sklearn.preprocessing import LabelEncoder  # nearest value will assume that it is related.
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.preprocessing import LabelBinarizer



	encoder_LB = LabelBinarizer()
	housing_cat_LB_1hot = encoder_LB.fit_transform(housing_cat)
	# print(housing_cat_LB_1hot)
	# print(housing.columns)
	   
	# attr_adder = FunctionTransformer(add_extra_features,validate = True, kw_args = {'add_bedroom_per_room':False})

	# housing_extra_attribs = attr_adder.fit_transform(housing.values)
	# print(housing.values)


	# housing_extra_attribs = pd.DataFrame(
 #    housing_extra_attribs,
 #    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
 #    index=housing.index)
	# housing_extra_attribs.head()
	# print (housing.columns)

	global rooms_ix,bedrooms_ix,population_ix,household_ix
	rooms_ix,bedrooms_ix,population_ix,household_ix = [list(housing.columns).index(col) for col in ("total_rooms","total_bedrooms","population","households")]
	# attr_adder = FunctionTransformer(add_extra_features, validate=False,kw_args={"add_bedrooms_per_room": False})
	# housing_extra_attribs = attr_adder.fit_transform(housing.values)

	from sklearn.preprocessing import FunctionTransformer

	def add_extra_features(X, add_bedrooms_per_room=True):
	    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
	    population_per_household = X[:, population_ix] / X[:, household_ix]
	    if add_bedrooms_per_room:
	        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
	        return np.c_[X, rooms_per_household, population_per_household,
	                     bedrooms_per_room]
	    else:
	        return np.c_[X, rooms_per_household, population_per_household]

	attr_adder = FunctionTransformer(add_extra_features, validate=False,
	                                 kw_args={"add_bedrooms_per_room": False})
	housing_extra_attribs = attr_adder.fit_transform(housing.values)



	housing_extra_attribs = pd.DataFrame(housing_extra_attribs,columns=list(housing.columns)+["rooms_per_household","population_per_household"],index = housing.index)
	# print(housing_extra_attribs.head(10))


	num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('attribs_adder',FunctionTransformer(add_extra_features,validate=False)),('std_scaler',StandardScaler()),])
	housing_num_tr = num_pipeline.fit_transform(housing_num)
	# print(housing_num_tr)

	from sklearn.compose import ColumnTransformer

	num_attribs = list(housing_num)
	cat_attribs = ["ocean_proximity"]
	full_pipeline = ColumnTransformer([("num",num_pipeline,num_attribs),
		("cat",OneHotEncoder(),cat_attribs),
		])
	housing_prepared = full_pipeline.fit_transform(housing)
	# print (housing_prepared)
	# print (housing_prepared.shape)
	# print (housing_labels.shape)

	from sklearn.linear_model import LinearRegression


	lin_reg = LinearRegression()
	lin_reg.fit(housing_prepared,housing_labels)


	# print (housing.iloc[:5])

	# print(housing_prepared.shape)

	some_data = housing.iloc[:5]
	some_labels = housing_labels.iloc[:5]
	# print(some_data.shape)
	
	some_data_prepared = full_pipeline.transform(some_data)
	print('prediction:',lin_reg.predict(some_data_prepared))
	print('Actual:',list(some_labels))



	from sklearn.metrics import mean_squared_error

	housing_predictions = lin_reg.predict(housing_prepared)
	lin_mse = mean_squared_error(housing_labels, housing_predictions)
	lin_rmse = np.sqrt(lin_mse)
	print(lin_rmse)


	from sklearn.metrics import mean_absolute_error

	lin_mae = mean_absolute_error(housing_labels, housing_predictions)
	print(lin_mae)


	from sklearn.tree import DecisionTreeRegressor

	tree_reg = DecisionTreeRegressor(random_state = 42)
	tree_reg.fit(housing_prepared,housing_labels)

	housing_predictions = tree_reg.predict(housing_prepared)
	tree_mse = mean_squared_error(housing_labels, housing_predictions)
	tree_rmse = np.sqrt(tree_mse)
	print(tree_rmse)




main()
