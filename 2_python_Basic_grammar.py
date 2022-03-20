#------------------------首先声明中文编码方式-------------
#-*-coding:utf-8-*-
#-------------------------如上一行即声明了中文编码方式-----


"""
Created on 2022年1月5日11:23:41
@author:yukuan zhang
第二章 python基础语法
"""


"""
1、输出语句
python的输出语句主要通过print（）实现。
print（）函数包括两种形式，print a， print（a）都可以输出a的变量
如果需要输出多个变量，则使用逗号连接，如print a，b，c
"""

x ,y ,c = 2,3,6
#print(x,y,c)
#----------------------输出-----------------------------
"""
2 3 6
"""

"""
2、通过format（val， format_modifier）函数实现格式化数据
其中，val表示值、format_modifier表示格式字段
6.2f表示输出六位数值的浮点数，小数点后精确到两位，输出值的最后一位采用四舍五入方式计算，最终输出的结果为12.35:2%表示输出百分数，保留两位有效数字，其输出结果为34.57%。如果想输出整数直接使用.0f即可。
"""

x = format(12.3456789, '6.2f')
y = format(12.3456789, '6.0f')
#print(x,y)

"""
3、变量命名规则
变量名是由大小写字符、数字和下换线组合而成的。
变量名的第一个字符必须是字母或下划线。
python中的变量是区分大小写的
在python中对变量进行赋值时，使用单引号和双引号是一样的效果。
python中的变量不需要声明，就可以直接使用赋值运算符对其进行赋值操作，根据所赋的值来决定其数据类型

python中的赋值语句是使用等号（=）给变量直接赋值
a = 1
a，c，b = 1,2,3
"""

"""
4、输入语句
python中的输入语句包括input（）、raw_input（）
input（）函数从控制台获取用户输入的字符串或值，并保存在变量中。
"""
#s = input("请输入")
# print(s,type(s))

"""
5、python中的数据类型
数字类型、字符串类型、列表类型、字典类型、元组类型等
数字类型：整数类型、返回类型是int。
        浮点数类型、带有小数点的数字，返回类型是float。
        复数类型、a+bj，返回类型为complex 方法.real获取实部、.imag获取虚部。
字符串类型：指需要用单引号或双引号括起来的一个字符或字符串。字符串表示一个字符的序列，其最左端表示字符串的起始位置，下标为0.使用len（）函数可以计算字符串长度。使用a+b可以将a与b所对应的字符串拼接
列表类型：在中括号[]中用逗号分隔的元素集合，列表中的元素可以通过索引(如a[2])进行单个访问。
字典类型：字典是针对非有序列集合而提供的，由键值对组成，如{key：value}。字典是键值对的集合，其类型为dict。键是字典的索引（如dic["key"]），一个键对应一个值，通过键值可以找到字典中的信息，这个过程叫做映射。
        使用dict.keys（）可获得字典中的键，使用dict.values（）可获得字典中的值
        直接给索引赋值可以在字典中增加或修改值。使用del dict['key']删除字典中的元素。使用dict.clear（）清空字典数据。使用del dict删除字典
元组类型：在小括号()中用逗号分隔的元素集合，其返回类型为tuple。可以通过索引访问元组中的某个元素（如a[2]）。当元组定义后就不能进行更改，也不能删除，元组的不可变特性使它的代码更加安全。
"""


"""
6、基本语句
常用的语句包括顺序语句、条件语句和循环语句
语句块：语句块并非一种语句，它是叜条件为真时执行一次或执行多次的一组语句，在代码前放置空格缩进即可创建语句块。
在python中使用 ：号表示语句块的开始。
块中的每一条语句都有缩进并且缩进量相同，当回到上一层缩进量时，就表示当前语句块已经结束。
"""
"""
条件语句：包括单分支、二分支、多分支三种情况
判断条件如果为真则执行语句块，否则跳过语句块。
条件判断通常有布尔表达式（True、False）、关系表达式（> < >= <= == !=）、逻辑运算表达式（and、or、not，其优先级从高到底是not、and、or） 

"""

#-------------单分支--------------
"""
a = 1
b = 2
c = 10
if a < b:
    print(a)
"""
#-------------二分支---------------
"""
if a >= 0:
    print(a)
else:
    print(b)
"""
#--------------多分支--------------
"""
if a < 0:
    print("大家好")
elif a < 1:
    print("你们好")
elif b <= c:
    print("我好")

"""
"""
循环语句：主要分为while循环和for循环
"""

#-----------while循环的基本格式，如果条件表达式为真，则循环体重复执行，否则终止循环。--------------
"""
nums = [1,2,3,4,5,6,7,8,9]
k = 0
while k < len(nums):
    print("大家新年快乐！")
    k += 1
else:
    print("新年过完了！")
"""
# --------------for循环语句的基本格式如下：自定义循环变量var便利sequence序列中的每一个值，每个值执行一次循环的语句块。sequence表示序列，常见类型有列表、元组、字符串、文件。----
"""
for <var> in <sequence>:
    <语句块>
<语句块>
"""
"""
for list in nums:
    print(list)
    list += list
    print("*" * list)
print("sum = ",list)
"""

"""
7、基本操作
自定义函数：
def funtion_name(para1,para2,...,paraN):
    statement1
    statement2
    ...
    return value1,value2,...,valueN
导入包：
import module_name
使用包：
module_name.method(parametes)
"""
