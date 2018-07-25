import numpy as np

a = [1,0.2,-0.021,0.323,0,-1,1,2,-0.9094,1,2,3,1,2,3,1,2,3,1,1,1,-0.1,0,100]
a = np.array(a).reshape(4,2,3) #[batch_size,data_len,embeding_size]
print 'stage1:\n',a,a.shape
a = np.abs(a)
print 'stage2:\n',a,a.shape
a = np.add(a,1)
print 'stage3:\n',a,a.shape
a = np.sign(a)
print 'stage4:\n',a,a.shape
a = np.sum(a,2)
print 'stage5:\n',a,a.shape
a = np.sign(a)
print 'stage6:\n',a,a.shape
a = np.sum(a,1)
print 'stage7:\n',a,a.shape