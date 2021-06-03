import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/car_price.csv')
print(data)

# 建立训练集
train_x = data[["symboling", "fueltype", "aspiration", "doornumber",
                "carbody", "drivewheel", "enginelocation", "wheelbase", "carlength", "carwidth",
                "carheight", "curbweight", "enginetype", "cylindernumber", "enginesize", "fuelsystem",
                "boreratio", "stroke", "compressionratio", "horsepower", "peakrpm", "citympg",
                "highwaympg", "price"]]
print(train_x)

# 将非数值字段转化为数值
le = LabelEncoder()
columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']
for column in columns:
    # LabelEncoder().transform将标签转换为归一化的编码。fit_transform安装标签编码器并返回编码的标签。
    train_x[column] = le.fit_transform(train_x[column])
print(train_x)

# 规范化到 [0,1] 空间
min_max_scaler = preprocessing.MinMaxScaler()
# MinMaxScaler()将每个要素缩放到给定范围，拟合数据，然后对其进行转换。
train_x = min_max_scaler.fit_transform(train_x)
pd.DataFrame(train_x).to_csv('temp.csv', index=False)

# 选择聚类组数
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)
    # 计算inertia簇内误差平方和
    sse.append(kmeans.inertia_)
x = range(1, 11)
# 将图像嵌入在结果中
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()

# 使用KMeans聚类,分成4类
kmeans = KMeans(n_clusters=4)
kmeans.fit(train_x)  # 也可以直接fit+predict
# predict计算聚类中心并预测每个样本的聚类索引。
predict_y = kmeans.predict(train_x)
print(predict_y)

# 合并聚类结果，插入到原数据中,axis： 需要合并链接的轴，0是行，1是列
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
# 将结果列重命名为'聚类结果'
result.rename({0: u'聚类结果'}, axis=1, inplace=True)
print(result)

# 找出vokswagen汽车的聚类结果
label = result[result.CarName.str.contains('vokswagen')]['聚类结果']
print(label)
List = ['vokswagen', 'volkswagen']
mask = result.CarName.str.contains('|'.join(List))
selected_data = result[mask]
print(selected_data)

# Vokswagen 轿车 竞品按价格排序
# lambda 是为了减少单行函数的定义而存在的,lambda作为一个表达式，定义了一个匿名函数，上例的代码x为入口参数，x['聚类结果']==3and 'sedan' in x['carbody']为函数体，
# 将函数应用到由各列或行形成的一维数组上。DataFrame的apply方法可以实现此功能。默认情况下会以列为单位,axis = 1以行为单位
# [[]]选择多列时用双括号
print(result[result.apply(lambda x: x['聚类结果'] == 3 and 'sedan' in x['carbody'], axis=1)][['CarName', "wheelbase", "price", 'horsepower', 'carbody', 'fueltype', '聚类结果']].sort_values('price', ascending=False))

# Vokswagen wagon 竞品按价格排序
print(result[result.apply(lambda x: x['聚类结果'] == 3 and 'wagon' in x['carbody'], axis=1)][['CarName', "wheelbase", "price", 'horsepower', 'carbody', 'fueltype', '聚类结果']].sort_values('price', ascending=False))

# 显示竞品车总体聚类结果
benchmark = result[result['聚类结果'] == 3].CarName
print("竞品车如下所示")
print(benchmark)
