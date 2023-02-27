# tryC - a small interpreter written by C

tryC is a very simple interpreter made by hand in C language, about 570 lines:

Use `recursive descent method` for grammatical analysis, do not explicitly build a grammar tree, generate intermediate code or target code; interpret and execute at the same time as grammatical analysis.

The try language implemented by tryC: 

- `Dynamic types`, supported data types: `double-precision floating-point numbers`, `character types`, `strings`, `arrays of floating-point numbers`;
- Support the definition of `functions` and `variables`, recursive call of functions, nested scope;

tryC是一个用c语言手搓的非常简单的解释器，大约570行：

采用递归下降法进行语法分析，不显式构建语法树，不生成中间代码或目标代码，在语法分析的同时进行解释执行；

tryC实现的try语言：
- 动态类型，支持的数据类型：双精度浮点数、字符型、字符串、浮点数数组
- 支持函数和变量的定义、函数的递归调用、嵌套作用域

### files：

source：tryC.c

example：test.try

### build and run：

build:

    gcc -o tryc tryC.c

usage: 
    
    tryc [-d] filename

## documents in Chinese

[用c语言手搓一个600行的类c语言解释器: 给编程初学者的解释器教程（1）- 目标和前言](https://blog.csdn.net/qq_42779423/article/details/105938297)

[用c语言手搓一个600行的类c语言解释器: 给编程初学者的解释器教程（2）- 简介和设计](https://blog.csdn.net/qq_42779423/article/details/105939788)

[用c语言手搓一个600行的类c语言解释器: 给编程初学者的解释器教程（3）- 词法分析](https://blog.csdn.net/qq_42779423/article/details/105948289)

[用c语言手搓一个600行的类c语言解释器: 给编程初学者的解释器教程（4）- 语法分析1：EBNF和递归下降文法](https://blog.csdn.net/qq_42779423/article/details/105954353)

[用c语言手搓一个600行的类c语言解释器: 给编程初学者的解释器教程（5）- 语法分析2: tryC的语法分析实现](https://blog.csdn.net/qq_42779423/article/details/105954455)

[用c语言手搓一个600行的类c语言解释器: 给编程初学者的解释器教程（6）- 语义分析：符号表和变量、函数](https://blog.csdn.net/qq_42779423/article/details/105979594)