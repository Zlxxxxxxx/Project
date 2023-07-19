import time
import pandas as pd
from sklearn.metrics import accuracy_score

from sklearn import tree
# import pydotplus
# from IPython.display import display, Image
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('SuddenChangeInBehaviourAlerts.csv')


feature_columns = ['f.monthly_avg_atm_trans_ct', 'f.daily_atm_tran_ct', 'f.monthly_avg_atm_trans_amt', 'f.daily_atm_tran_amt']
label_columns = ['f.acct_risk']

features = data[feature_columns]
label = data[label_columns]

x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.4)

start = time.time()

dtree = tree.DecisionTreeClassifier(criterion='gini')

dtree = dtree.fit(x_train.values, y_train.values)

y_train_pred = dtree.predict(x_train.values)
y_test_pred = dtree.predict(x_test.values)

# rfc = RandomForestClassifier(criterion='gini')
# rfc = rfc.fit(x_train.values, y_train.values.ravel())

# y_train_pred = rfc.predict(x_train.values)
# y_test_pred = rfc.predict(x_test.values)


print("Training sets："+str(accuracy_score(y_train_pred,y_train.values.ravel())))
print("Testing sets："+str(accuracy_score(y_test_pred,y_test.values.ravel())))

end = time.time()
print ("执行时间："+str(end-start))

_dict = dict(zip(features.columns.tolist(),dtree.feature_importances_))
_dict = sorted( _dict.items()  ,key=lambda x:x[1],reverse=True )
print(_dict)

# dot_data = tree.export_graphviz(
#     rfc,
#     out_file=None,
#     feature_names=feature_columns,
#     class_names=['Low', 'Medium', 'High'],
#     rounded=True,
#     filled=True
# )

# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("C:/Users/10489/Desktop/dtree.pdf")
