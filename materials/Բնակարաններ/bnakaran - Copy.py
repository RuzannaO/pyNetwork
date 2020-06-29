from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from plotter import plot_on_formula
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
# from sklearn.externals import joblib

def show_plot(df, field):
    y = df['price']
    x = df[field]
    plt.scatter(x, y)
    plt.title(field)
    print(plt.show())

class bnakaran:

    def __init__(self,filename):
        name_dict = {
            "district": ["str"], "condition": ["str"], "building_type": ["str"], 'street': ["str"],
            'price': ["int", [0, 5000000]], 'num_rooms': ["int", [0, 30]], 'area': ["int", [0, 500]],
            'num_bathrooms': ["int", [0, 10]],"max_floor":["int",[0,150]],"floor":["int",[0,150]],"ceiling_height":["float",[2,5]]}

        self.dataset=pd.read_csv(filename,encoding='ISO-8859-1')
        ##-- validate headers names
        if set(self.dataset.columns.values)!=set(['Unnamed: 0',"url",'region','price','condition','district','max_floor','street','num_rooms','area','num_bathrooms','building_type','floor','ceiling_height']):
            raise ValueError("INCORRECT SET OF HEADERS NAMES")
        self.dataset=self.dataset.drop(columns=['Unnamed: 0', 'region','url'])
        #check headers' and values' accuracy
        for i in list(self.dataset.columns.values):
            if i not in name_dict.keys():
                raise ValueError(f'Field name "{i}" not correct!')
            for j in self.dataset[i]:
                if not isinstance(j,eval(name_dict[i][0])):
                      raise TypeError (f'Incorrect type in Field "{i}"",  must be {name_dict[i][0]}! ')
                if not isinstance(j,str):
                    if j> name_dict[i][1][1] or j< name_dict[i][1][0]:
                        raise ValueError(f'Incorrect value in the Field "{i}", {j}, must be in range {name_dict[i][1]}! ')



    def describe(self):
        return(self.dataset.describe().to_string())
    def preprocessing(self):
        # removing duplicates
        # self.subset = ['price', 'condition', 'district', 'max_floor', 'street', 'num_rooms', 'area', 'num_bathrooms',
        #                'num_bathrooms', 'floor', 'ceiling_height']
        # y = self.dataset.drop_duplicates(subset=self.subset, keep='first', inplace=True)
        #normalizing "streets',"districts","building_type","condition"
            ##-- creates (reads) street - coefficients dict
        df = pd.read_csv("Streets.csv", engine='python')
        condition_dict={"newly repaired": 1, "good": 0.85, "zero condition": 0.7}
        building_type_dict={"monolit":1,"stone":2,"other":3,'panel':4}
        street_dict=pd.Series(df.Street_koef.values, index=df.street).to_dict()
            ##-- creates (reads) district - coefficients dict
        df1 = pd.read_csv("Districts.csv", engine='python')
        district_dict=pd.Series(df1.district_coef.values, index=df1.district).to_dict()
        ##-- if not available , add to the corresponding dict with value
        #**********************************************************************
        make_changes={"condition":0.8,"building_type":2,"district":0.45}
        for field_name in ["condition","building_type","district"]:
            for i in self.dataset[field_name]:
                if i not in eval(f'{field_name}_dict'):

                    eval(f'{field_name}_dict')[str(i)] = make_changes[str(field_name)]
            print("eval",eval(f'{field_name}_dict'))


            ## -- creates a dict (district: street) grouped by district
        distr_str_dic=self.dataset.groupby('district').agg({'street': lambda x: x.tolist()})['street'].to_dict()
        # if new names in streets' list, then makes corresponding changes in corr. dicts

        strt_list=(self.dataset['street'].tolist())
        for i in self.dataset.street.values:
            if i not in street_dict.keys():
                for k,v in distr_str_dic.items():
                    if i not in v:
                        street_dict[i]=district_dict[k]
            ##--validate district,condition, buiding type values
        if not set(self.dataset['district'].tolist()).issubset(set([*district_dict])):
            raise ValueError ("INCORRECT VALUE DISTRICT NAME!!!")
        name_dict={"street":street_dict, "district": district_dict,
                     "condition":condition_dict,"building_type":building_type_dict}
        for j in ["building_type","condition",'district','street']:
                self.dataset[j]=self.dataset[j].map(name_dict[j])
        self.dataset.to_csv("output3.csv")
        # mycol_train.insert_many(self.dataset.to_dict("records"))
        return self.dataset
    def fit_multi_lin(self,df,simple=0):
        labels = df['price']
        if simple==1:
            train1 = df.drop(['price', 'district', 'num_bathrooms', "max_floor", "ceiling_height",'condition', 'street', 'num_rooms','building_type', 'floor'],
                             axis=1)
        else:
            train1 = df.drop(['price', 'district','num_bathrooms', "max_floor","ceiling_height"],
                         axis=1)
        x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.1, random_state=1)
        reg = LinearRegression(fit_intercept=True).fit(train1, labels)
        self.mylist=train1.keys().tolist()
        print(self.mylist)
        reg.fit(x_train, y_train)
        self.intercept=reg.intercept_
        self.coef=reg.coef_
        predictions=reg.predict(x_test)
        plt.scatter(y_test,predictions)
        plt.show()
        print(len(predictions))
        print("Here is the accuracy check on the train data")
        print('MAE:', metrics.mean_absolute_error(y_test, predictions))
        print('MSE:', metrics.mean_squared_error(y_test, predictions))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
        return self.intercept,self.coef,self.mylist
    def plotting_on_formula(self,mylist):
        # creates a graph on predicted linear formula(arguments:intersept and coefficents)
        # print("Here is the input data that the graph will be constructed on")
        # print("intercept:",self.intercept,"coefficients:", self.coef, "list of features:",self.mylist,"your input:",mylist)
        # yy = []
        # y = intercept
        # for k in mylist:
        #      y+= k[0] * b[0] + k[1] * b[1]
        #     yy.append(y)
        # plt.plot(mylist, yy)
        # print(mylist, yy)
        return plt.show()


        plot_on_formula(self.intercept, self.coef, mylist)
        return
def predict(df,intc,coef,mylist):
    Y_test = df["price"]
    df.insert(1,"predictions",'')
    print(my_list)
    for i in range(0, len(df)):
        y = intc
        for l in range(0,len(my_list)):
            y = y+coef[l]*df[str(my_list[l])][i]
        df["predictions"][i]=y
    print("Here is the accuracy check on the Test data")
    print('MAE:', metrics.mean_absolute_error(df['price'], df['predictions']))
    print('MSE:', metrics.mean_squared_error(df['price'], df['predictions']))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(df['price'], df['predictions'])))
    # print("ADDITIONALLY:   HERE IS THE PRICE INFO ON AVAILABLE RECORDS IN OUR DATABASE")
    # df.drop("price", axis=1, inplace=True)
    # df.rename(columns={"predictions":"price_calculated"},inplace=True)
    # for i in df.to_dict("records"):
    #     print(i)
    #     rec=i.pop("price_calculated")
    #     n=mycol_train.find_one(i,{'_id':0})
    #     if n:
    #         print(n)
    #         print("---------------------------------------------------------")


    df.to_excel("output5.xlsx")
    return df.head(50)

# myclient=MongoClient('localhost',27017)
# mydb=myclient['Houses']
# mycol_predict=mydb["Houses"]
# mycol_train=mydb["Houses"]
# mycol_predict.drop()
# mycol_train.drop()


x=bnakaran('houses_train (1).csv')
y=x.preprocessing()
# mycol_train.insert_many(y.to_dict("records"))
# for j in mycol_train.find():
#     s=s+1
#     print(s,j)
print(x.fit_multi_lin(y))
y=bnakaran("houses_train (1).csv")
m=y.preprocessing()
intc,coef,my_list=x.fit_multi_lin(m,1)

predict(m, intc,coef,my_list)
print("*******************************************************************")


# test_dict=y.dataset.to_dict("records")
#
# # test_dict=mylist={"_id":3,"price":40000,"street":0.1,"floor":3 }


x.plotting_on_formula([[1,100],[1,150],[1,200]])
df=pd.DataFrame(x.dataset)
# print(df.head().to_string())
# print(show_plot(df,"street"))
# print(show_plot(df,"district"))
# show_plot(x.dataset,"num_rooms")
# print(show_plot(df,"area"))
# print(show_plot(df,"max_floor"))
# print(show_plot(df,"condition"))
# print(show_plot(df,"building_type"))
# print(show_plot(df,"floor"))
# print(show_plot(df,"ceiling_height"))
# print(show_plot(df,"num_bathrooms"))




# df1=m.head(100)
# df1.plot(kind='bar',figsize=(16,10))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()


# train1 = m.drop(['price','num_rooms', 'building_type','district','num_bathrooms', "max_floor", "floor", "max_floor", "ceiling_height", 'condition'],
#                          axis=1)
# labels = m['price']
# x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.5, random_state=0)
#
#
# clf=ensemble.GradientBoostingClassifier(n_estimators=1000,max_features=0.1,max_depth=5,min_samples_leaf=2,learning_rate=0.1)
# print(clf.fit(x_train,y_train))
# print(clf.score(x_test,y_test))
#











# y=x.dataset[['district','street']]
# z=y.drop_duplicates(keep='first')
# z.sort_values(by='district',inplace=True)
# x.dataset.to_excel('output1.xlsx')

# slope, intercept, r, p, std_err = stats.linregress(x, y)
# print(r)
# print(slope)
# print(intercept)
# str=stats.linregress(x,y)
# print(str)
