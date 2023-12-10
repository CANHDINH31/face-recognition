import numpy as np

# sample = ['Red', 'Green', 'Blue']
#
p = [0.2, 0.45, 0.35]
q = [0.31, 0.25, 0.44]
#
# entropy_p = -sum([p[i] * np.log(p[i]) for i in range(len(p))])
# entropy_q = -sum([q[i] * np.log(q[i]) for i in range(len(q))])
#
# print(entropy_p)
# print(entropy_q)
#
#
crossentropy_pq = -sum([p[i] * np.log(q[i]) for i in range(len(p))])
crossentropy_qp = -sum([q[i] * np.log(p[i]) for i in range(len(q))])
#
# print(crossentropy_pq)
# print(crossentropy_qp)
#
#
# klDivergence_pq = sum([p[i] * np.log(p[i]/q[i]) for i in range(len(p))])
# print(klDivergence_pq)

y = [0.1, 0.1, 0.2, 0.3, 0.1, 0.2]
yhat = [0.111, 0.44, 0.122, 0.3, 0.121, 0.2234 ]

cross_entropy_y_yhat = -sum([y[i] * np.log(yhat[i]) for i in range(len(y))])
cross_entropy_y_yhat =  np.log(yhat[i])


