#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import matplotlib.image as mpimg
import scipy.misc
import numpy as np
import os
import glob
from multiprocessing import Process

src_dir= "/project/EvolvingAI/mnorouzz/Serengiti/SER/S6/"
dst_dir= "/gscratch/mnorouzz/S6/"


def divide(t,n,i):
    length=t/(n+0.0)
    #print length,(i-1)*length,i*length
    return int(round((i-1)*length)),int(round(i*length))

def do_chunk(pid,filelist):
    for co,row in enumerate(filelist):
      try:
        if co%1000==0:
          print(co)
          sys.stdout.flush()
        img=mpimg.imread(row)
        img=scipy.misc.imresize(img[0:-100,:],(256,256))
        path=dst_dir+str(row[len(src_dir):row.rfind('/')])

        if not os.path.exists(path):
          os.makedirs(path)
        mpimg.imsave(dst_dir+row[len(src_dir):],img)
      except:
        print 'Severe Error for'+row
        #raise
    print("Process "+str(pid)+"  is done")

if __name__ == '__main__':
    try:
      allfiles=[]
      for path, subdirs, files in os.walk(src_dir):
        for f in files:
          if f.endswith(".JPG") or f.endswith(".jpg"):
            allfiles.append(os.path.join(path,f))
            if len(allfiles)%10000==0:
              print(len(allfiles))
      total_records=len(allfiles)
      total_processors=16
      print(total_records)
      for i in range(1,total_processors+1):
        st,ln=divide(total_records,total_processors,i)
        p1 = Process(target=do_chunk, args=(i,allfiles[st:ln]))
        p1.start()
    except:
      raise
