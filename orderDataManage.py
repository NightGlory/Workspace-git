import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(
    font_scale=1.5,
    style="whitegrid",
    rc={'figure.figsize': (20, 7)}
)

# 载入数据
sales_team = pd.read_csv('data/sales_team.csv')
order_leads = pd.read_csv('data/order_leads.csv')
invoices = pd.read_csv('data/invoices.csv')

# 对order_leads.Date使用pd.DateTimeIndex，将其设置为序号;
# 使用pd.grouped（freq='D'）按天对数据进行分组;
_ = order_leads.set_index(pd.DatetimeIndex(order_leads.Date)).groupby(
    pd.Grouper(freq='D')
)['Converted'].mean()       # 计算每天“转化”的平均值，即当天订单的转化率

# 使用.rolling（60）和.mean（）得到60天的平均值
ax = _.rolling(60).mean().plot(figsize=(20, 7),
                               title='Conversion Rate Over Time')

vals = ax.get_yticks()
# 设置yticklables的格式，使其显示百分比符号
ax.set_yticklabels(['{:,.0f}%'.format(x*100) for x in vals])
sns.despine()

# 不同销售代表的转化率
orders_with_sales_team = pd.merge(order_leads, sales_team, on=[
                                  'Company Id', 'Company Name'])
ax = sns.distplot(orders_with_sales_team.groupby(
    'Sales Rep Id')['Converted'].mean(), kde=False)
vals = ax.get_xticks()
ax.set_xticklabels(['{:,.0f}%'.format(x*100) for x in vals])
ax.set_title('Number of sales reps by conversion rate')
sns.despine()

# 销售团队的数据
# 把垂直线映射到每个子块中，并用数据的平均值和标准偏差来注释这条线


def vertical_mean_line(x, **kwargs):
    ls = {"0": "-", "1": "--"}
    plt.axvline(x.mean(), linestyle=ls[kwargs.get("label", "0")],
                color=kwargs.get("color", "r"))
    txkw = dict(size=15, color=kwargs.get("color", "r"))
    tx = "mean: {:.1f}%\n(std: {:.1f}%)".format(x.mean()*100, x.std()*100)
    label_x_pos_adjustment = 0.015
    label_y_pos_adjustment = 20
    plt.text(x.mean() + label_x_pos_adjustment,
             label_y_pos_adjustment, tx, **txkw)


sns.set(
    font_scale=1.5,
    style="whitegrid"
)

_ = orders_with_sales_team.groupby('Sales Rep Id').agg({
    'Converted': np.mean,
    'Company Id': pd.Series.nunique
})
_.columns = ['conversion rate', 'number of accounts']

g = sns.FacetGrid(_, col="number of accounts",
                  height=4, aspect=0.9, col_wrap=5)
g.map(sns.kdeplot, "conversion rate", shade=True)
g.set(xlim=(0, 0.35))
g.map(vertical_mean_line, "conversion rate")

# 快速查看时间的分布
invoices['Date of Meal'] = pd.to_datetime(invoices['Date of Meal'])
invoices['Date of Meal'].dt.time.value_counts().sort_index()

# 根据早中晚餐分割数据集
invoices['Date of Meal'] = pd.to_datetime(invoices['Date of Meal'])
invoices['Type of Meal'] = pd.cut(
    invoices['Date of Meal'].dt.hour,
    bins=[0, 10, 15, 24],
    labels=['breakfast', 'lunch', 'dinner']
)

# 把第一个字符串invoices['Participants']转换成合法的JSON，这样可以提取参与者的数量


def replace(x):
    return x.replace("\n ", ",").replace("' '", "','").replace("'", '"')


invoices['Participants'] = invoices['Participants'].apply(lambda x: replace(x))
invoices['Number Participants'] = invoices['Participants'].apply(
    lambda x: len(json.loads(x)))

# 左结合 order_leads 和 invoice data 的数据
orders_with_invoices = pd.merge(
    order_leads, invoices, how='left', on='Company Id')

# 计算用餐和订单之间的时间差
orders_with_invoices['Days of meal before order'] = (
    pd.to_datetime(orders_with_invoices['Date']) -
    orders_with_invoices['Date of Meal']
).dt.days

# 限制只考虑订单前后5天的用餐
orders_with_invoices = orders_with_invoices[abs(
    orders_with_invoices['Days of meal before order']) < 5]

# 仍有一些订单匹配了多个用餐信息。这可能发生在同时有两个订单也有两次用餐的情况。两个订单线索都会匹配两次用餐
# 为了去掉那些重复数据，我们只保留与订单时间最接近的那个订单
orders_with_invoices = orders_with_invoices.loc[
    abs(orders_with_invoices['Days of meal before order']).sort_values().index
]

# 保留第一个（即最接近销售事件）销售订单
orders_with_invoices = orders_with_invoices.drop_duplicates(subset=[
                                                            'Order Id'])

orders_without_invoices = order_leads[~order_leads['Order Id'].isin(
    orders_with_invoices['Order Id'].unique())]

orders_with_meals = pd.concat(
    [orders_with_invoices, orders_without_invoices], sort=True)

# 创建了一个柱状图函数，展示用餐种类的影响


def plot_bars(data, x_col, y_col):
    data = data.reset_index()
    sns.set(
        font_scale=1.5,
        style="whitegrid",
        rc={'figure.figsize': (20, 7)}
    )
    g = sns.barplot(x=x_col, y=y_col, data=data, color='royalblue')

    for p in g.patches:
        g.annotate(
            format(p.get_height(), '.2%'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 10),
            textcoords='offset points'
        )

    vals = g.get_yticks()
    g.set_yticklabels(['{:,.0f}%'.format(x*100) for x in vals])

    sns.despine()


# 有没有用餐信息对订单转化率有明显的差别
orders_with_meals['Type of Meal'].fillna('no meal', inplace=True)
_ = orders_with_meals.groupby('Type of Meal').agg({'Converted': np.mean})
plot_bars(_, x_col='Type of Meal', y_col='Converted')

# “用餐在订单前几天发生”为负数意味着用餐是在订单线索出现之后
# 用餐时机的影响
_ = orders_with_meals.groupby(['Days of meal before order']).agg(
    {'Converted': np.mean}
)
plot_bars(data=_, x_col='Days of meal before order', y_col='Converted')

# 使用热图同时显示数据的多个维度。为此我们先创建一个函数


def draw_heatmap(data, inner_row, inner_col, outer_row, outer_col, values):
    sns.set(font_scale=1)
    fg = sns.FacetGrid(
        data,
        row=outer_row,
        col=outer_col,
        margin_titles=True
    )

    position = left, bottom, width, height = 1.4, .2, .1, .6
    cbar_ax = fg.fig.add_axes(position)

    fg.map_dataframe(
        draw_heatmap_facet,
        x_col=inner_col,
        y_col=inner_row,
        values=values,
        cbar_ax=cbar_ax,
        vmin=0,
        vmax=.4
    )

    fg.fig.subplots_adjust(right=1.3)
    plt.show()


def draw_heatmap_facet(*args, **kwargs):
    data = kwargs.pop('data')
    x_col = kwargs.pop('x_col')
    y_col = kwargs.pop('y_col')
    values = kwargs.pop('values')
    d = data.pivot(index=y_col, columns=x_col, values=values)
    annot = round(d, 4).values
    cmap = sns.color_palette("RdYlGn", 30)
    # cmap = sns.color_palette("PuBu",30) alternative color coding
    sns.heatmap(d, **kwargs, annot=annot, center=0,
                fmt=".1%", cmap=cmap, linewidth=.5)


# 应用一些数据处理来探索用餐花销与订单价值的关系，并将我们的用餐时间划分为订单前（Before Order）、订单前后（Around Order）、订单后（After Order），
# 而不是从负4到正4的天数，因为这解读起来会比较麻烦
orders_with_meals['Meal Price / Order Value'] = orders_with_meals['Meal Price'] / \
    orders_with_meals['Order Value']
orders_with_meals['Meal Price / Order Value'] = pd.qcut(
    orders_with_meals['Meal Price / Order Value']*-1,
    5,
    labels=['Least Expensive', 'Less Expensive', 'Proportional',
            'More Expensive', 'Most Expensive'][::-1]
)

orders_with_meals['Timing of Meal'] = pd.qcut(
    orders_with_meals['Days of meal before order'],
    3,
    labels=['After Order', 'Around Order', 'Before Order']
)

data = orders_with_meals[orders_with_meals['Type of Meal'] != 'no meal'].groupby(
    ['Timing of Meal', 'Number Participants',
        'Type of Meal', 'Meal Price / Order Value']
).agg({'Converted': np.mean}).unstack().fillna(0).stack().reset_index()

# 运行下面的代码片段将生成多维热图
draw_heatmap(
    data=data,
    outer_row='Timing of Meal',
    outer_col='Type of Meal',
    inner_row='Meal Price / Order Value',
    inner_col='Number Participants',
    values='Converted'
)

# 解读这张热图
"""
图表总结了4个不同维度的影响：

用餐时间：订单后、订单前后、订单前（外行）
用餐类型：早餐、晚餐、午餐（外列）
餐费/订单价值：最低、较低、成比例、较贵、最贵（内行）
参加人数：1,2,3,4,5（内列）

当然，看起来图表底部的颜色更暗/更高，这表明

在订单前用餐的转化率更高
似乎晚餐的转换率更高，当只有一个人用餐时
看起来相对订单价值，更贵的餐费对转化率有积极的影响

结果：

1.不要给你的销售代表超过9个客户（因为转化率下降很快）；
2.确保每个订单线索都有会议/用餐（因为这使转化率翻倍）；
3.当只分配一名员工给顾客时，晚餐是最有效的；
4.你的销售代表应该支付大约为订单价值8%到10%的餐费；
5.时机是关键，理想情况下，让你的销售代表尽早知道交易即将达成。
"""
