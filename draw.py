# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2023/3/22
# # @Author  : hehl
# # @Software: PyCharm
# @File    : draw.py

import matplotlib.pyplot as plt
import numpy as np

x = [0.1, 0.3, 0.5, 0.7, 0.9]
y_1 = [74.34, 76.66, 77.23, 75.74, 74.59]
y_2 = [71.88, 74.41, 75.29, 74.63, 71.32]
y_3 = [74.71, 76.68, 77.47, 76.04, 75.43]
y_4 = [72.58, 74.89, 76.36, 74.38, 72.42]
y_5 = [72.40, 73.26, 75.98, 73.48, 72.23]
# y2 = [5966 / 1639, 9044 / 1639, 10390 / 1639, 11729 / 1639, 13866 / 1639, 16370 / 1639, 19846 / 1639, 21389 / 1639]

fig = plt.figure(figsize=(28, 10))
plt.rcParams.update({'font.size': 30})

ax = fig.add_subplot(121)
ax.set_xticks(x)
ax.set_yticks(range(71, 79))

lns1 = ax.plot(x, y_1, '-', label=r'$k$=1')
lns2 = ax.plot(x, y_2, '-', label=r'$k$=2')
lns3 = ax.plot(x, y_3, '-', label=r'$k$=3')
lns4 = ax.plot(x, y_4, '-', label=r'$k$=4')
lns5 = ax.plot(x, y_5, '-', label=r'$k$=5')

# ax2 = ax.twinx()
# lns3 = ax2.plot(x, y2, '--r', linewidth=2, label=r'Avg-$G$')

# added these three lines
lns = lns1 + lns2 + lns3 + lns4 + lns5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=8)

# 画出最高点
ax.scatter(0.5, 77.47, s=100, facecolors='r', edgecolors='r')
# 注释位置
ax.annotate(77.50, xy=(0.52, 77.47))

ax.grid()
ax.set_xlabel(r"(a) varied $k$ and $\xi$ on WebQuestionSP")
ax.set_ylabel(r"Performance evaluation(F1%)")
# ax2.set_ylabel(r"Average number of query graphs")
# ax2.set_ylim(0, 14)
ax.set_ylim(71, 78)
#


cq_1 = [45.57, 46.00, 47.34, 46.33, 45.31]
cq_2 = [44.88, 45.64, 46.59, 45.64, 44.82]
cq_3 = [43.97, 45.04, 46.52, 45.43, 44.08]
cq_4 = [43.84, 45.28, 46.11, 45.32, 44.02]
cq_5 = [43.80, 44.53, 45.52, 44.97, 43.39]
# cqy2 = [2777 / 800, 4797 / 800, 5315 / 800, 6043 / 800, 6724 / 800, 8333 / 800, 9742 / 800, 10619 / 800]

cq = fig.add_subplot(122)
cq.set_xticks(x)
cq.set_yticks(range(43, 49))

cq_lns1 = cq.plot(x, cq_1, '-', label=r'$k$=1')
cq_lns2 = cq.plot(x, cq_2, '-', label=r'$k$=2')
cq_lns3 = cq.plot(x, cq_3, '-', label=r'$k$=3')
cq_lns4 = cq.plot(x, cq_4, '-', label=r'$k$=4')
cq_lns5 = cq.plot(x, cq_5, '-', label=r'$k$=5')

# cq2 = cq.twinx()
# cq_lns3 = cq2.plot(x, cqy2, '--r', linewidth=2, label=r'Avg-$G$')

cqlns = cq_lns1 + cq_lns2 + cq_lns3 + cq_lns4 + cq_lns5
cqlabs = [l.get_label() for l in cqlns]
cq.legend(cqlns, cqlabs, loc=8)

cq.scatter(0.5, 47.34, s=100, facecolors='r', edgecolors='r')
cq.annotate(47.4, xy=(0.52, 47.34))

cq.grid()
cq.set_xlabel(r"(b) varied $k$ and $\xi$ on ComplexQuestions")
cq.set_ylabel(r"Performance evaluation(F1%)")
# cq2.set_ylabel(r"Average number of query graphs")
# cq2.set_ylim(0, 14)
cq.set_ylim(43, 48)



plt.subplots_adjust(wspace=0.25, top=0.96, bottom=0.1, left=0.05, right=0.95)
# plt.subplots_adjust()

foo_fig = plt.gcf()  # 'get current figure'
foo_fig.savefig('F1_hyperpara.eps', format='eps', dpi=600)
plt.show()
