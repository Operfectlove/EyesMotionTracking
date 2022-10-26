import datetime 
global before1 = datetime.datetime.now()
f = open('tf.txt','w')
def work(x):
    now1 = datetime.datetime.now()
    ddt= now1 - before1
    if ddt.seconds >= 5:
        f.write(x)
        x =''
        ddt = datetime.datetime.now()
    else:
        global before1 = datetime.datetime.now()

