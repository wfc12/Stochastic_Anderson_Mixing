import pickle
with open('output.ser','rb') as f:
    res = pickle.load(f)
ans = res['valprec_avg']
print(ans.index(max(ans)),max(ans))
#for i,v in enumerate(ans):
#    print(i,v)

testprec,testloss = res['testprec'],res['testloss']
print('testprec:',testprec)
print('testloss:',testloss)
