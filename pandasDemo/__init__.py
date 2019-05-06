# -----------------------------
# pandas基础数据结构
# -----------------------------
import numpy as np
import pandas as pd

s = pd.Series([i*2 for i in range(1, 11)])
print(type(s))

dates = pd.date_range("20190507", periods=8)
df = pd.DataFrame(np.random.rand(8, 5), index=dates, columns=list("ABCDE"))
print(df)

# 另一种形式
df = pd.DataFrame({"A": 1, "B":pd.Timestamp("20190507"), "C":pd.Series(1, index=list(range(4)), dtype="float32"), \
                    "D":np.array([3]*4, dtype="float32"), "E":pd.Categorical(["ploice", "student", "teacher", "doctor"])})
print(df)


# -----------------------------
# pandas基础操作
# -----------------------------
dates = pd.date_range("20190507", periods=8)
df = pd.DataFrame(np.random.rand(8, 5), index=dates, columns=list("ABCDE"))

# 打印前三行
print(df.head(3))
# 打印后几行
print(df.tail(3))
# 打印索引
print(df.index)
# 打印值
print(df.values)
# 转置
print(df.T)
# 排序
print(df.sort_index(axis=1, ascending=False))
print(df.sort_values("A"))  # 实际对应B，因为不包含索引
# 打印均值，标准差，最小值等统计数据
print(df.describe())

# select
print(df["A"])
print(df[:3])
print(df["20190507":"20190509"])
print(df.loc[dates[0]])
print(df.loc["20190507":"20190509", ["B","D"]])
print(df.at[dates[0], "C"])

# 通过下标选择
print(df.iloc[1:3, 2:4])
print(df.iloc[1,4])
print(df.iat[1,4])

# 填入条件
print(df[df.B>0][df.A>0])
print(df[df>0])
print(df[df["E"].isin([1,2])])

# 增删改查Set
# 增加
sl = pd.Series(list(range(10, 18)), index=pd.date_range("20190507", periods=8))
df["F"]=sl
print(df)
# 修改
df.at[dates[0], "A"]=0
print(df)
df.iat[1,1]=1
df.loc[:,"D"]=np.array([4*len(df)])
print(df)
# 复制数据
df2 = df.copy()
df2[df2>0]=-df2
print(df2)


# -----------------------------
# pandas缺失值处理
# -----------------------------
df1 = df.reindex(index=dates[:4], columns=list("ABCD") + ["G"])
df1.loc[dates[0]:dates[1], "G"]=1
print(df1)

# 缺失值丢弃处理
print(df1.dropna())

# 缺失值填充固定值处理
print(df1.fillna(value=1))

# 缺失值差值处理，见上节


# -----------------------------
# pandas表统计与整合
# -----------------------------
print(df.mean())
print(df.var())

s = pd.Series([1,2,4,np.nan, 5,7,9,10], index=dates)
print(s)
print(s.shift(2))       # 表前列补空值
print(s.diff())         # 与前一个数据的差值
print(s.value_counts()) # 每一个值在Series中出现的次数

print(df.apply(np.cumsum))  # 累加
# 自定义函数
print(df.apply(lambda x:x.max()-x.min()))

# 拼接
pieces=[df[:3], df[-3:]]
print(pd.concat(pieces))

# join 方式
left = pd.DataFrame({"key":["x", "y"], "value":[1,2]})
right = pd.DataFrame({"key":["x", "z"], "value":[3,4]})
print("left: ", left)

print(pd.merge(left, right, on="key", how="outer"))

# groupby
df3 = pd.DataFrame({"A": ["a", "b", "c", "b"], "B": list(range(4))})
print(df3.groupby("A").sum())

# reshape透视表
import datetime
df4 = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
                    'B': ['a', 'b', 'c'] * 8,
                    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
                    'D': np.random.randn(24), 
                    'E': np.random.randn(24),
                    'F': [datetime.datetime(2019, i, 1) for i in range(1, 13)]+ 
                        [datetime.datetime(2019, i, 15) for i in range(1, 13)]})

print(pd.pivot_table(df4, values="D", index=["A", "B"], columns=["C"]))


# -----------------------------
# pandas时间、绘图、文件操作
# -----------------------------
# 时间序列
t_exam = pd.date_range("20190507", periods=10, freq="S")
print(t_exam)

# 绘图
ts = pd.Series(np.random.randn(1000), index = pd.date_range("20190507", periods=1000))
ts = ts.cumsum()

from pylab import *
ts.plot()
show()

# 文件操作
df6 = pd.read_excel("pandasDemo/test.xlsx", "Sheet1")
print(df6)

# 保存
df6.to_csv("./pandasDemo/save.csv")