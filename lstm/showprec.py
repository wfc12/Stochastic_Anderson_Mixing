import pickle
with open('output.ser','rb') as f:
    res = pickle.load(f)
ans = res['valppl']
print(ans.index(min(ans)),min(ans))
#for i,v in enumerate(ans):
#    print(i,v)

testppl,testloss = res['testppl'],res['testloss']
print('testppl:',testppl)
print('testloss:',testloss)
